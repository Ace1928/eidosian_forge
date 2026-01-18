import pyomo.environ as pyo
def ScensIterator(self):
    """Usage: for scenario in ScensIterator()"""
    return iter(self._scens)