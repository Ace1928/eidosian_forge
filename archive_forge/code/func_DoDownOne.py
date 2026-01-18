from . import screen
from . import FSM
import string
def DoDownOne(fsm):
    screen = fsm.memory[0]
    screen.cursor_down()