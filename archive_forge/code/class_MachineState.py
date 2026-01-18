from enum import Enum, Flag
class MachineState:
    """
    This enum represents the different states a state machine can be in.
    """
    START = 0
    ERROR = 1
    ITS_ME = 2