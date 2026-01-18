from . import screen
from . import FSM
import string
def DoEraseDown(fsm):
    screen = fsm.memory[0]
    screen.erase_down()