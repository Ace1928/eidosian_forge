import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
def _create_delete_access_rule(self, share_id, access_type, access_to, microversion=None):
    self.skip_if_microversion_not_supported(microversion)
    if access_type not in self.access_types:
        raise self.skipException("'%(access_type)s' access rules is disabled for protocol '%(protocol)s'." % {'access_type': access_type, 'protocol': self.protocol})
    access = self.user_client.access_allow(share_id, access_type, access_to, self.access_level, microversion=microversion)
    self.assertEqual(share_id, access.get('share_id'))
    self.assertEqual(access_type, access.get('access_type'))
    self.assertEqual(access_to.replace('\\\\', '\\'), access.get('access_to'))
    self.assertEqual(self.access_level, access.get('access_level'))
    if api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.33'):
        self.assertIn('access_key', access)
        self.assertIn('created_at', access)
        self.assertIn('updated_at', access)
    elif api_versions.APIVersion(microversion) >= api_versions.APIVersion('2.21'):
        self.assertIn('access_key', access)
    else:
        self.assertNotIn('access_key', access)
    self.user_client.wait_for_access_rule_status(share_id, access['id'])
    self.user_client.access_deny(share_id, access['id'])
    self.user_client.wait_for_access_rule_deletion(share_id, access['id'])
    self.assertRaises(tempest_lib_exc.NotFound, self.user_client.get_access, share_id, access['id'])