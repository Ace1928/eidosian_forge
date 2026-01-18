from . import screen
from . import FSM
import string
def DoForward(fsm):
    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_forward(count)