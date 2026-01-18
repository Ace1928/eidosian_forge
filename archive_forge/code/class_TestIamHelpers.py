from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@SkipForS3('Tests use GS IAM model.')
@SkipForXML('XML IAM control is not supported.')
class TestIamHelpers(testcase.GsUtilUnitTestCase):
    """Unit tests for iam command helper."""

    def test_convert_bindings_simple(self):
        """Tests that Policy.bindings lists are converted to dicts properly."""
        self.assertEqual(BindingsMessageToUpdateDict([]), defaultdict(set))
        expected = defaultdict(set, {'x': set(['y'])})
        self.assertEqual(BindingsMessageToUpdateDict([bvle(role='x', members=['y'])]), expected)

    def test_convert_bindings_duplicates(self):
        """Test that role and member duplication are converted correctly."""
        expected = defaultdict(set, {'x': set(['y', 'z'])})
        duplicate_roles = [bvle(role='x', members=['y']), bvle(role='x', members=['z'])]
        duplicate_members = [bvle(role='x', members=['z', 'y']), bvle(role='x', members=['z'])]
        self.assertEqual(BindingsMessageToUpdateDict(duplicate_roles), expected)
        self.assertEqual(BindingsMessageToUpdateDict(duplicate_members), expected)

    def test_convert_bindings_dict_simple(self):
        """Tests that Policy.bindings lists are converted to dicts properly."""
        self.assertEqual(BindingsDictToUpdateDict([]), defaultdict(set))
        expected = defaultdict(set, {'x': set(['y'])})
        self.assertEqual(BindingsDictToUpdateDict([{'role': 'x', 'members': ['y']}]), expected)

    def test_convert_bindings_dict_duplicates(self):
        """Test that role and member duplication are converted correctly."""
        expected = defaultdict(set, {'x': set(['y', 'z'])})
        duplicate_roles = [{'role': 'x', 'members': ['y']}, {'role': 'x', 'members': ['z']}]
        duplicate_members = [{'role': 'x', 'members': ['z', 'y']}, {'role': 'x', 'members': ['z']}]
        self.assertEqual(BindingsDictToUpdateDict(duplicate_roles), expected)
        self.assertEqual(BindingsDictToUpdateDict(duplicate_members), expected)

    def test_equality_bindings_literal(self):
        """Tests an easy case of identical bindings."""
        bindings = [bvle(role='x', members=['y'])]
        self.assertTrue(IsEqualBindings([], []))
        self.assertTrue(IsEqualBindings(bindings, bindings))

    def test_equality_bindings_extra_roles(self):
        """Tests bindings equality when duplicate roles are added."""
        bindings = [bvle(role='x', members=['x', 'y'])]
        bindings2 = bindings * 2
        bindings3 = [bvle(role='x', members=['y']), bvle(role='x', members=['x'])]
        self.assertTrue(IsEqualBindings(bindings, bindings2))
        self.assertTrue(IsEqualBindings(bindings, bindings3))

    def test_diff_bindings_add_role(self):
        """Tests simple grant behavior of Policy.bindings diff."""
        expected = [bvle(role='x', members=['y'])]
        granted, removed = DiffBindings([], expected)
        self.assertEqual(granted.bindings, expected)
        self.assertEqual(removed.bindings, [])

    def test_diff_bindings_drop_role(self):
        """Tests simple remove behavior of Policy.bindings diff."""
        expected = [bvle(role='x', members=['y'])]
        granted, removed = DiffBindings(expected, [])
        self.assertEqual(granted.bindings, [])
        self.assertEqual(removed.bindings, expected)

    def test_diff_bindings_swap_role(self):
        """Tests expected behavior of switching a role."""
        old = [bvle(role='x', members=['y'])]
        new = [bvle(role='a', members=['b'])]
        granted, removed = DiffBindings(old, new)
        self.assertEqual(granted.bindings, new)
        self.assertEqual(removed.bindings, old)

    def test_diff_bindings_add_member(self):
        """Tests expected behavior of adding a member to a role."""
        old = [bvle(role='x', members=['y'])]
        new = [bvle(role='x', members=['z', 'y'])]
        expected = [bvle(role='x', members=['z'])]
        granted, removed = DiffBindings(old, new)
        self.assertEqual(granted.bindings, expected)
        self.assertEqual(removed.bindings, [])

    def test_diff_bindings_drop_member(self):
        """Tests expected behavior of dropping a member from a role."""
        old = [bvle(role='x', members=['z', 'y'])]
        new = [bvle(role='x', members=['y'])]
        expected = [bvle(role='x', members=['z'])]
        granted, removed = DiffBindings(old, new)
        self.assertEqual(granted.bindings, [])
        self.assertEqual(removed.bindings, expected)

    def test_diff_bindings_swap_member(self):
        """Tests expected behavior of switching a member in a role."""
        old = [bvle(role='x', members=['z'])]
        new = [bvle(role='x', members=['y'])]
        granted, removed = DiffBindings(old, new)
        self.assertEqual(granted.bindings, new)
        self.assertEqual(removed.bindings, old)

    def test_patch_bindings_grant(self):
        """Tests patching a grant binding."""
        base_list = [bvle(role='a', members=['user:foo@bar.com']), bvle(role='b', members=['user:foo@bar.com']), bvle(role='c', members=['user:foo@bar.com'])]
        base = BindingsMessageToUpdateDict(base_list)
        diff_list = [bvle(role='d', members=['user:foo@bar.com'])]
        diff = BindingsMessageToUpdateDict(diff_list)
        expected = BindingsMessageToUpdateDict(base_list + diff_list)
        res = PatchBindings(base, diff, True)
        self.assertEqual(res, expected)

    def test_patch_bindings_remove(self):
        """Tests patching a remove binding."""
        base = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
        diff = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a')])
        expected = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
        res = PatchBindings(base, diff, False)
        self.assertEqual(res, expected)

    def test_patch_bindings_remove_all(self):
        """Tests removing all roles from a member."""
        base = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
        diff = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='')])
        res = PatchBindings(base, diff, False)
        self.assertEqual(res, {})
        diff = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
        res = PatchBindings(base, diff, False)
        self.assertEqual(res, {})

    def test_patch_bindings_multiple_users(self):
        """Tests expected behavior when multiple users exist."""
        expected = BindingsMessageToUpdateDict([bvle(members=['user:fii@bar.com'], role='b')])
        base = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com', 'user:fii@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
        diff = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
        res = PatchBindings(base, diff, False)
        self.assertEqual(res, expected)

    def test_patch_bindings_grant_all_users(self):
        """Tests a public member grant."""
        base = BindingsMessageToUpdateDict([bvle(role='a', members=['user:foo@bar.com']), bvle(role='b', members=['user:foo@bar.com']), bvle(role='c', members=['user:foo@bar.com'])])
        diff = BindingsMessageToUpdateDict([bvle(role='a', members=['allUsers'])])
        expected = BindingsMessageToUpdateDict([bvle(role='a', members=['allUsers', 'user:foo@bar.com']), bvle(role='b', members=['user:foo@bar.com']), bvle(role='c', members=['user:foo@bar.com'])])
        res = PatchBindings(base, diff, True)
        self.assertEqual(res, expected)

    def test_patch_bindings_public_member_overwrite(self):
        """Tests public member vs. public member interaction."""
        base_list = [bvle(role='a', members=['allUsers'])]
        base = BindingsMessageToUpdateDict(base_list)
        diff_list = [bvle(role='a', members=['allAuthenticatedUsers'])]
        diff = BindingsMessageToUpdateDict(diff_list)
        res = PatchBindings(base, diff, True)
        self.assertEqual(res, BindingsMessageToUpdateDict(base_list + diff_list))

    def test_valid_public_member_single_role(self):
        """Tests parsing single role (case insensitive)."""
        _, bindings = bstt(True, 'allusers:admin')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.admin'}, bindings)

    def test_grant_no_role_error(self):
        """Tests that an error is raised when no role is specified for a grant."""
        with self.assertRaises(CommandException):
            bstt(True, 'allUsers')
        with self.assertRaises(CommandException):
            bstt(True, 'user:foo@bar.com')
        with self.assertRaises(CommandException):
            bstt(True, 'user:foo@bar.com:')
        with self.assertRaises(CommandException):
            bstt(True, 'deleted:user:foo@bar.com?uid=1234:')

    def test_remove_all_roles(self):
        """Tests parsing a -d allUsers or -d user:foo@bar.com request."""
        is_grant, bindings = bstt(False, 'allUsers')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['allUsers'], 'role': ''}, bindings)
        self.assertEqual((is_grant, bindings), bstt(False, 'allUsers:'))
        _, bindings = bstt(False, 'user:foo@bar.com')
        self.assertEqual(len(bindings), 1)

    def test_valid_multiple_roles(self):
        """Tests parsing of multiple roles bound to one user."""
        _, bindings = bstt(True, 'allUsers:a,b,c,roles/custom')
        self.assertEqual(len(bindings), 4)
        self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.a'}, bindings)
        self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.b'}, bindings)
        self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.c'}, bindings)
        self.assertIn({'members': ['allUsers'], 'role': 'roles/custom'}, bindings)

    def test_valid_custom_roles(self):
        """Tests parsing of custom roles bound to one user."""
        _, bindings = bstt(True, 'user:foo@bar.com:roles/custom1,roles/custom2')
        self.assertEqual(len(bindings), 2)
        self.assertIn({'members': ['user:foo@bar.com'], 'role': 'roles/custom1'}, bindings)
        self.assertIn({'members': ['user:foo@bar.com'], 'role': 'roles/custom2'}, bindings)

    def test_valid_member(self):
        """Tests member parsing (case insensitive)."""
        _, bindings = bstt(True, 'User:foo@bar.com:admin')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['user:foo@bar.com'], 'role': 'roles/storage.admin'}, bindings)

    def test_valid_deleted_member(self):
        """Tests deleted member parsing (case insensitive)."""
        _, bindings = bstt(False, 'Deleted:User:foo@bar.com?uid=123')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['deleted:user:foo@bar.com?uid=123'], 'role': ''}, bindings)
        _, bindings = bstt(True, 'deleted:User:foo@bar.com?uid=123:admin')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['deleted:user:foo@bar.com?uid=123'], 'role': 'roles/storage.admin'}, bindings)
        _, bindings = bstt(True, 'deleted:user:foo@bar.com?query=param,uid=123?uid=456:admin,admin2')
        self.assertEqual(len(bindings), 2)
        self.assertIn({'members': ['deleted:user:foo@bar.com?query=param,uid=123?uid=456'], 'role': 'roles/storage.admin'}, bindings)
        self.assertIn({'members': ['deleted:user:foo@bar.com?query=param,uid=123?uid=456'], 'role': 'roles/storage.admin2'}, bindings)

    def test_duplicate_roles(self):
        """Tests that duplicate roles are ignored."""
        _, bindings = bstt(True, 'allUsers:a,a')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['allUsers'], 'role': 'roles/storage.a'}, bindings)

    def test_removing_project_convenience_groups(self):
        """Tests that project convenience roles can be removed."""
        _, bindings = bstt(False, 'projectViewer:123424:admin')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['projectViewer:123424'], 'role': 'roles/storage.admin'}, bindings)
        _, bindings = bstt(False, 'projectViewer:123424')
        self.assertEqual(len(bindings), 1)
        self.assertIn({'members': ['projectViewer:123424'], 'role': ''}, bindings)

    def test_adding_project_convenience_groups(self):
        """Tests that project convenience roles cannot be added."""
        with self.assertRaises(CommandException):
            bstt(True, 'projectViewer:123424:admin')

    def test_invalid_input(self):
        """Tests invalid input handling."""
        with self.assertRaises(CommandException):
            bstt(True, 'non_valid_public_member:role')
        with self.assertRaises(CommandException):
            bstt(True, 'non_valid_type:id:role')
        with self.assertRaises(CommandException):
            bstt(True, 'user:r')
        with self.assertRaises(CommandException):
            bstt(True, 'deleted:user')
        with self.assertRaises(CommandException):
            bstt(True, 'deleted:not_a_type')
        with self.assertRaises(CommandException):
            bstt(True, 'deleted:user:foo@no_uid_suffix')

    def test_invalid_n_args(self):
        """Tests invalid input due to too many colons."""
        with self.assertRaises(CommandException):
            bstt(True, 'allUsers:some_id:some_role')
        with self.assertRaises(CommandException):
            bstt(True, 'user:foo@bar.com:r:nonsense')
        with self.assertRaises(CommandException):
            bstt(True, 'deleted:user:foo@bar.com?uid=1234:r:nonsense')