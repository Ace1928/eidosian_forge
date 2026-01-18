from . import screen
from . import FSM
import string
def DoEraseEndOfLine(fsm):
    screen = fsm.memory[0]
    screen.erase_end_of_line()