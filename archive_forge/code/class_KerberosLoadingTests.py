from keystoneauth1 import loading
from keystoneauth1.tests.unit import utils as test_utils
class KerberosLoadingTests(test_utils.TestCase):

    def test_options(self):
        opts = [o.name for o in loading.get_plugin_loader('v3kerberos').get_options()]
        allowed_opts = ['system-scope', 'domain-id', 'domain-name', 'project-id', 'project-name', 'project-domain-id', 'project-domain-name', 'trust-id', 'auth-url', 'mutual-auth']
        self.assertCountEqual(allowed_opts, opts)