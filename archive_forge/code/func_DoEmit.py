from . import screen
from . import FSM
import string
def DoEmit(fsm):
    screen = fsm.memory[0]
    screen.write_ch(fsm.input_symbol)