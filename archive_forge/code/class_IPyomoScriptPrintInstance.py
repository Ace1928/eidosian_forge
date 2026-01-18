from pyomo.common.plugin_base import (
class IPyomoScriptPrintInstance(Interface):

    def apply(self, **kwds):
        """Apply instance printing step in the Pyomo script"""