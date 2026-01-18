from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
class TestBranchOpenerCheckAndFollowBranchReference(TestCase):
    """Unit tests for `BranchOpener.check_and_follow_branch_reference`."""

    def setUp(self):
        super().setUp()
        BranchOpener.install_hook()

    class StubbedBranchOpener(BranchOpener):
        """BranchOpener that provides canned answers.

        We implement the methods we need to to be able to control all the
        inputs to the `follow_reference` method, which is what is
        being tested in this class.
        """

        def __init__(self, references, policy):
            parent_cls = TestBranchOpenerCheckAndFollowBranchReference
            super(parent_cls.StubbedBranchOpener, self).__init__(policy)
            self._reference_values = {}
            for i in range(len(references) - 1):
                self._reference_values[references[i]] = references[i + 1]
            self.follow_reference_calls = []

        def follow_reference(self, url):
            self.follow_reference_calls.append(url)
            return self._reference_values[url]

    def make_branch_opener(self, should_follow_references, references, unsafe_urls=None):
        policy = _BlacklistPolicy(should_follow_references, unsafe_urls)
        opener = self.StubbedBranchOpener(references, policy)
        return opener

    def test_check_initial_url(self):
        opener = self.make_branch_opener(None, [], {'a'})
        self.assertRaises(BadUrl, opener.check_and_follow_branch_reference, 'a')

    def test_not_reference(self):
        opener = self.make_branch_opener(False, ['a', None])
        self.assertEqual('a', opener.check_and_follow_branch_reference('a'))
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_branch_reference_forbidden(self):
        opener = self.make_branch_opener(False, ['a', 'b'])
        self.assertRaises(BranchReferenceForbidden, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_allowed_reference(self):
        opener = self.make_branch_opener(True, ['a', 'b', None])
        self.assertEqual('b', opener.check_and_follow_branch_reference('a'))
        self.assertEqual(['a', 'b'], opener.follow_reference_calls)

    def test_check_referenced_urls(self):
        opener = self.make_branch_opener(True, ['a', 'b', None], unsafe_urls=set('b'))
        self.assertRaises(BadUrl, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_self_referencing_branch(self):
        opener = self.make_branch_opener(True, ['a', 'a'])
        self.assertRaises(BranchLoopError, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a'], opener.follow_reference_calls)

    def test_branch_reference_loop(self):
        references = ['a', 'b', 'a']
        opener = self.make_branch_opener(True, references)
        self.assertRaises(BranchLoopError, opener.check_and_follow_branch_reference, 'a')
        self.assertEqual(['a', 'b'], opener.follow_reference_calls)