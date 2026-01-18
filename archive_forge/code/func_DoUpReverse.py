from . import screen
from . import FSM
import string
def DoUpReverse(fsm):
    screen = fsm.memory[0]
    screen.cursor_up_reverse()