from . import screen
from . import FSM
import string
def DoBack(fsm):
    count = int(fsm.memory.pop())
    screen = fsm.memory[0]
    screen.cursor_back(count)