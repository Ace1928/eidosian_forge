from . import screen
from . import FSM
import string
def DoDown(fsm):
    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_down(count)