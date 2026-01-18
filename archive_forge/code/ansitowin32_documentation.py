import re
import sys
import os
from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test

        Write the given text to our wrapped stream, stripping any ANSI
        sequences from the text, and optionally converting them into win32
        calls.
        