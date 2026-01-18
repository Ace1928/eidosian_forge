from . import screen
from . import FSM
import string
def DoCursorSave(fsm):
    screen = fsm.memory[0]
    screen.cursor_save_attrs()