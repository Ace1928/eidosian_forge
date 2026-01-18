from . import screen
from . import FSM
import string
def DoForwardOne(fsm):
    screen = fsm.memory[0]
    screen.cursor_forward()