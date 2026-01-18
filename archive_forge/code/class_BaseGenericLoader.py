from keystoneauth1 import exceptions
from keystoneauth1.loading import base
from keystoneauth1.loading import opts
class BaseGenericLoader(BaseIdentityLoader):
    """Base Option handling for generic plugins.

    This class defines options and handling that should be common to generic
    plugins. These plugins target the OpenStack identity service however are
    designed to be independent of API version. It provides the options expected
    by the :py:class:`keystoneauth1.identity.v3.BaseGenericPlugin` class.
    """

    def get_options(self):
        options = super(BaseGenericLoader, self).get_options()
        options.extend([opts.Opt('system-scope', help='Scope for system operations'), opts.Opt('domain-id', help='Domain ID to scope to'), opts.Opt('domain-name', help='Domain name to scope to'), opts.Opt('project-id', help='Project ID to scope to', deprecated=[opts.Opt('tenant-id')]), opts.Opt('project-name', help='Project name to scope to', deprecated=[opts.Opt('tenant-name')]), opts.Opt('project-domain-id', help='Domain ID containing project'), opts.Opt('project-domain-name', help='Domain name containing project'), opts.Opt('trust-id', help='ID of the trust to use as a trustee use'), opts.Opt('default-domain-id', help='Optional domain ID to use with v3 and v2 parameters. It will be used for both the user and project domain in v3 and ignored in v2 authentication.'), opts.Opt('default-domain-name', help='Optional domain name to use with v3 API and v2 parameters. It will be used for both the user and project domain in v3 and ignored in v2 authentication.')])
        return options