import sys, os, unicodedata
import py
from py.builtin import text, bytes
class Win32ConsoleWriter(TerminalWriter):

    def write(self, msg, **kw):
        if msg:
            if not isinstance(msg, (bytes, text)):
                msg = text(msg)
            self._update_chars_on_current_line(msg)
            oldcolors = None
            if self.hasmarkup and kw:
                handle = GetStdHandle(STD_OUTPUT_HANDLE)
                oldcolors = GetConsoleInfo(handle).wAttributes
                default_bg = oldcolors & 240
                attr = default_bg
                if kw.pop('bold', False):
                    attr |= FOREGROUND_INTENSITY
                if kw.pop('red', False):
                    attr |= FOREGROUND_RED
                elif kw.pop('blue', False):
                    attr |= FOREGROUND_BLUE
                elif kw.pop('green', False):
                    attr |= FOREGROUND_GREEN
                elif kw.pop('yellow', False):
                    attr |= FOREGROUND_GREEN | FOREGROUND_RED
                else:
                    attr |= oldcolors & 7
                SetConsoleTextAttribute(handle, attr)
            write_out(self._file, msg)
            if oldcolors:
                SetConsoleTextAttribute(handle, oldcolors)