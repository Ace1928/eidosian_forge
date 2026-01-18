from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
def _add_common_identity_options(options):
    options.extend([loading.Opt('user-id', help='User ID'), loading.Opt('username', help='Username', deprecated=[loading.Opt('user-name')]), loading.Opt('user-domain-id', help="User's domain id"), loading.Opt('user-domain-name', help="User's domain name")])