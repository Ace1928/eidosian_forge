import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
class PickPolicyFileTestCase(base.PolicyBaseTestCase):

    def setUp(self):
        super(PickPolicyFileTestCase, self).setUp()
        self.data = {'rule_admin': 'True', 'rule_admin2': 'is_admin:True'}
        self.tmpdir = self.useFixture(fixtures.TempDir())
        original_search_dirs = cfg._search_dirs

        def fake_search_dirs(dirs, name):
            dirs.append(self.tmpdir.path)
            return original_search_dirs(dirs, name)
        mock_search_dir = self.useFixture(fixtures.MockPatch('oslo_config.cfg._search_dirs')).mock
        mock_search_dir.side_effect = fake_search_dirs
        mock_cfg_location = self.useFixture(fixtures.MockPatchObject(self.conf, 'get_location')).mock
        mock_cfg_location.return_value = cfg.LocationInfo(cfg.Locations.set_default, 'None')

    def test_no_fallback_to_json_file(self):
        tmpfilename = 'policy.yaml'
        self.conf.set_override('policy_file', tmpfilename, group='oslo_policy')
        jsonfile = os.path.join(self.tmpdir.path, 'policy.json')
        with open(jsonfile, 'w') as fh:
            jsonutils.dump(self.data, fh)
        selected_policy_file = policy.pick_default_policy_file(self.conf, fallback_to_json_file=False)
        self.assertEqual(self.conf.oslo_policy.policy_file, tmpfilename)
        self.assertEqual(selected_policy_file, tmpfilename)

    def test_overridden_policy_file(self):
        tmpfilename = 'nova-policy.yaml'
        self.conf.set_override('policy_file', tmpfilename, group='oslo_policy')
        selected_policy_file = policy.pick_default_policy_file(self.conf)
        self.assertEqual(self.conf.oslo_policy.policy_file, tmpfilename)
        self.assertEqual(selected_policy_file, tmpfilename)

    def test_only_new_default_policy_file_exist(self):
        self.conf.set_override('policy_file', 'policy.yaml', group='oslo_policy')
        tmpfilename = os.path.join(self.tmpdir.path, 'policy.yaml')
        with open(tmpfilename, 'w') as fh:
            yaml.dump(self.data, fh)
        selected_policy_file = policy.pick_default_policy_file(self.conf)
        self.assertEqual(self.conf.oslo_policy.policy_file, 'policy.yaml')
        self.assertEqual(selected_policy_file, 'policy.yaml')

    def test_only_old_default_policy_file_exist(self):
        self.conf.set_override('policy_file', 'policy.yaml', group='oslo_policy')
        tmpfilename = os.path.join(self.tmpdir.path, 'policy.json')
        with open(tmpfilename, 'w') as fh:
            jsonutils.dump(self.data, fh)
        selected_policy_file = policy.pick_default_policy_file(self.conf)
        self.assertEqual(self.conf.oslo_policy.policy_file, 'policy.yaml')
        self.assertEqual(selected_policy_file, 'policy.json')

    def test_both_default_policy_file_exist(self):
        self.conf.set_override('policy_file', 'policy.yaml', group='oslo_policy')
        tmpfilename1 = os.path.join(self.tmpdir.path, 'policy.json')
        with open(tmpfilename1, 'w') as fh:
            jsonutils.dump(self.data, fh)
        tmpfilename2 = os.path.join(self.tmpdir.path, 'policy.yaml')
        with open(tmpfilename2, 'w') as fh:
            yaml.dump(self.data, fh)
        selected_policy_file = policy.pick_default_policy_file(self.conf)
        self.assertEqual(self.conf.oslo_policy.policy_file, 'policy.yaml')
        self.assertEqual(selected_policy_file, 'policy.yaml')