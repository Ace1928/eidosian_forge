import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
class BeautifierFlags:

    def __init__(self, mode):
        self.mode = mode
        self.parent = None
        self.last_token = Token(TOKEN.START_BLOCK, '')
        self.last_word = ''
        self.declaration_statement = False
        self.declaration_assignment = False
        self.multiline_frame = False
        self.inline_frame = False
        self.if_block = False
        self.else_block = False
        self.class_start_block = False
        self.do_block = False
        self.do_while = False
        self.import_block = False
        self.in_case = False
        self.in_case_statement = False
        self.case_body = False
        self.case_block = False
        self.indentation_level = 0
        self.alignment = 0
        self.line_indent_level = 0
        self.start_line_index = 0
        self.ternary_depth = 0

    def apply_base(self, flags_base, added_newline):
        next_indent_level = flags_base.indentation_level
        if not added_newline and flags_base.line_indent_level > next_indent_level:
            next_indent_level = flags_base.line_indent_level
        self.parent = flags_base
        self.last_token = flags_base.last_token
        self.last_word = flags_base.last_word
        self.indentation_level = next_indent_level