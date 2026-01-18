from pyomo.common.plugin_base import (
class IPyomoScriptSaveResults(Interface):

    def apply(self, **kwds):
        """Apply results saving step in the Pyomo script"""