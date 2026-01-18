from keystoneauth1 import exceptions
from keystoneauth1.loading import base
from keystoneauth1.loading import opts
class BaseFederationLoader(BaseV3Loader):
    """Base Option handling for federation plugins.

    This class defines options and handling that should be common to the V3
    identity federation API. It provides the options expected by the
    :py:class:`keystoneauth1.identity.v3.FederationBaseAuth` class.
    """

    def get_options(self):
        options = super(BaseFederationLoader, self).get_options()
        options.extend([opts.Opt('identity-provider', help="Identity Provider's name", required=True), opts.Opt('protocol', help='Protocol for federated plugin', required=True)])
        return options