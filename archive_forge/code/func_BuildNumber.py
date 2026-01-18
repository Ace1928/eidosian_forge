import sys
import string
def BuildNumber(fsm):
    s = fsm.memory.pop()
    s = s + fsm.input_symbol
    fsm.memory.append(s)