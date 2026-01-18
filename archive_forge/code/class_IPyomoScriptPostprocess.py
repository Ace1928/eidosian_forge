from pyomo.common.plugin_base import (
class IPyomoScriptPostprocess(Interface):

    def apply(self, **kwds):
        """Apply postprocessing step in the Pyomo script"""