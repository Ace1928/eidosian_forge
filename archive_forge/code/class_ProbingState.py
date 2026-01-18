from enum import Enum, Flag
class ProbingState(Enum):
    """
    This enum represents the different states a prober can be in.
    """
    DETECTING = 0
    FOUND_IT = 1
    NOT_ME = 2