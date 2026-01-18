from suds import *
from logging import getLogger
class InitPlugin(Plugin):
    """Base class for all suds I{init} plugins."""

    def initialized(self, context):
        """
        Suds client initialization.

        Called after WSDL the has been loaded. Provides the plugin with the
        opportunity to inspect/modify the WSDL.

        @param context: The init context.
        @type context: L{InitContext}

        """
        pass