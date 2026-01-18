import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
def _do_password_change_for_one_user(self, password, new_password):
    self.config_fixture.config(group='ldap', use_auth_pool=True)
    self.cleanup_pools()
    self.load_backends()
    user1 = self._create_user_and_authenticate(password)
    auth_cm = self._get_auth_conn_pool_cm()
    self.assertEqual(1, len(auth_cm))
    user2 = self._create_user_and_authenticate(password)
    self.assertEqual(1, len(auth_cm))
    user3 = self._create_user_and_authenticate(password)
    self.assertEqual(1, len(auth_cm))
    user4 = self._create_user_and_authenticate(password)
    self.assertEqual(1, len(auth_cm))
    user5 = self._create_user_and_authenticate(password)
    self.assertEqual(1, len(auth_cm))
    user_api = ldap.UserApi(CONF)
    u1_dn = user_api._id_to_dn_string(user1['id'])
    u2_dn = user_api._id_to_dn_string(user2['id'])
    u3_dn = user_api._id_to_dn_string(user3['id'])
    u4_dn = user_api._id_to_dn_string(user4['id'])
    u5_dn = user_api._id_to_dn_string(user5['id'])
    auth_cm = self._get_auth_conn_pool_cm()
    with auth_cm.connection(u1_dn, password) as _:
        with auth_cm.connection(u2_dn, password) as _:
            with auth_cm.connection(u3_dn, password) as _:
                with auth_cm.connection(u4_dn, password) as _:
                    with auth_cm.connection(u5_dn, password) as _:
                        self.assertEqual(5, len(auth_cm))
                        _.unbind_s()
    user3['password'] = new_password
    PROVIDERS.identity_api.update_user(user3['id'], user3)
    return user3