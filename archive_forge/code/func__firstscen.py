import pyomo.environ as pyo
def _firstscen(self):
    assert len(self._scens) > 0
    return self._scens[0]