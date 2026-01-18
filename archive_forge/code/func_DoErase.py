from . import screen
from . import FSM
import string
def DoErase(fsm):
    arg = int(fsm.memory.pop())
    screen = fsm.memory[0]
    if arg == 0:
        screen.erase_down()
    elif arg == 1:
        screen.erase_up()
    elif arg == 2:
        screen.erase_screen()