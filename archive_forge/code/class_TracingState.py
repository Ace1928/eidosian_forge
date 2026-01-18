from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class TracingState(str, enum.Enum):
    ON = 'ON'
    OFF = 'OFF'