import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
class CallFrameInstruction(object):
    """ An instruction in the CFI section. opcode is the instruction
        opcode, numeric - as it appears in the section. args is a list of
        arguments (including arguments embedded in the low bits of some
        instructions, when applicable), decoded from the stream.
    """

    def __init__(self, opcode, args):
        self.opcode = opcode
        self.args = args

    def __repr__(self):
        return '%s (0x%x): %s' % (instruction_name(self.opcode), self.opcode, self.args)