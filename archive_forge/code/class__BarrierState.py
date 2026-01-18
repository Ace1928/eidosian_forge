import collections
import enum
from . import exceptions
from . import mixins
from . import tasks
class _BarrierState(enum.Enum):
    FILLING = 'filling'
    DRAINING = 'draining'
    RESETTING = 'resetting'
    BROKEN = 'broken'