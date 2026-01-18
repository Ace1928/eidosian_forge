import stevedore
from testtools import matchers
from keystonemiddleware.auth_token import _opts as new_opts
from keystonemiddleware import opts as old_opts
from keystonemiddleware.tests.unit import utils
def _test_list_auth_token_opts(self, result):
    self.assertThat(result, matchers.HasLength(1))
    for group in (g for g, _l in result):
        self.assertEqual('keystone_authtoken', group)
    expected_opt_names = ['www_authenticate_uri', 'interface', 'auth_uri', 'auth_version', 'delay_auth_decision', 'http_connect_timeout', 'http_request_max_retries', 'cache', 'certfile', 'keyfile', 'cafile', 'region_name', 'insecure', 'memcached_servers', 'token_cache_time', 'memcache_security_strategy', 'memcache_secret_key', 'memcache_use_advanced_pool', 'memcache_pool_dead_retry', 'memcache_pool_maxsize', 'memcache_pool_unused_timeout', 'memcache_pool_conn_get_timeout', 'memcache_pool_socket_timeout', 'include_service_catalog', 'enforce_token_bind', 'auth_type', 'auth_section', 'service_token_roles', 'service_token_roles_required', 'service_type']
    opt_names = [o.name for g, l in result for o in l]
    self.assertThat(opt_names, matchers.HasLength(len(expected_opt_names)))
    for opt in opt_names:
        self.assertIn(opt, expected_opt_names)