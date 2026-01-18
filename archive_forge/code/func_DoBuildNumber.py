from . import screen
from . import FSM
import string
def DoBuildNumber(fsm):
    ns = fsm.memory.pop()
    ns = ns + fsm.input_symbol
    fsm.memory.append(ns)