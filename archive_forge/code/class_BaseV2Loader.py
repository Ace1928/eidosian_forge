from keystoneauth1 import exceptions
from keystoneauth1.loading import base
from keystoneauth1.loading import opts
class BaseV2Loader(BaseIdentityLoader):
    """Base Option handling for identity plugins.

    This class defines options and handling that should be common to the V2
    identity API. It provides the options expected by the
    :py:class:`keystoneauth1.identity.v2.Auth` class.
    """

    def get_options(self):
        options = super(BaseV2Loader, self).get_options()
        options.extend([opts.Opt('tenant-id', help='Tenant ID'), opts.Opt('tenant-name', help='Tenant Name'), opts.Opt('trust-id', help='ID of the trust to use as a trustee use')])
        return options