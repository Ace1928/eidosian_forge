from . import screen
from . import FSM
import string
def DoEraseLine(fsm):
    arg = int(fsm.memory.pop())
    screen = fsm.memory[0]
    if arg == 0:
        screen.erase_end_of_line()
    elif arg == 1:
        screen.erase_start_of_line()
    elif arg == 2:
        screen.erase_line()