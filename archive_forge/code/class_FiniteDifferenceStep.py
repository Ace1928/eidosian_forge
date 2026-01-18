import pickle
from enum import Enum
from collections import namedtuple
class FiniteDifferenceStep(Enum):
    forward = 'forward'
    central = 'central'
    backward = 'backward'