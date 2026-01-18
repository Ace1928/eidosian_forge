import pyomo.environ as pyo
def ScenarioNumber(self, scennum):
    """Returns the scenario with the given, zero-based number"""
    return self._scens[scennum]