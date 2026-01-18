import enum
from pyomo.opt.results.container import MapContainer
class ProblemSense(str, enum.Enum):
    unknown = 'unknown'
    minimize = 'minimize'
    maximize = 'maximize'

    def __str__(self):
        return self.value