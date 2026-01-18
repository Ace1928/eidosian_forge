from . import screen
from . import FSM
import string
def DoEnableScroll(fsm):
    screen = fsm.memory[0]
    screen.scroll_screen()