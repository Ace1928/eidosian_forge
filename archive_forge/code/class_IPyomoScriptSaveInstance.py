from pyomo.common.plugin_base import (
class IPyomoScriptSaveInstance(Interface):

    def apply(self, **kwds):
        """Apply instance saving step in the Pyomo script"""