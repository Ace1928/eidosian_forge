from pyomo.common.plugin_base import (
class IPyomoPresolveAction(Interface):

    def presolve(self, instance):
        """Apply the presolve action to this instance, and return the
        revised instance"""

    def rank(self):
        """Return an integer that is used to automatically order presolve actions,
        from low to high rank."""