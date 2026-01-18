import enum
from pyomo.opt.results.container import MapContainer, ScalarType
class SolverStatus(str, enum.Enum):
    ok = 'ok'
    warning = 'warning'
    error = 'error'
    aborted = 'aborted'
    unknown = 'unknown'

    def __str__(self):
        return self.value