from . import win32
class WinTerm(object):

    def __init__(self):
        self._default = win32.GetConsoleScreenBufferInfo(win32.STDOUT).wAttributes
        self.set_attrs(self._default)
        self._default_fore = self._fore
        self._default_back = self._back
        self._default_style = self._style

    def get_attrs(self):
        return self._fore + self._back * 16 + self._style

    def set_attrs(self, value):
        self._fore = value & 7
        self._back = value >> 4 & 7
        self._style = value & WinStyle.BRIGHT

    def reset_all(self, on_stderr=None):
        self.set_attrs(self._default)
        self.set_console(attrs=self._default)

    def fore(self, fore=None, on_stderr=False):
        if fore is None:
            fore = self._default_fore
        self._fore = fore
        self.set_console(on_stderr=on_stderr)

    def back(self, back=None, on_stderr=False):
        if back is None:
            back = self._default_back
        self._back = back
        self.set_console(on_stderr=on_stderr)

    def style(self, style=None, on_stderr=False):
        if style is None:
            style = self._default_style
        self._style = style
        self.set_console(on_stderr=on_stderr)

    def set_console(self, attrs=None, on_stderr=False):
        if attrs is None:
            attrs = self.get_attrs()
        handle = win32.STDOUT
        if on_stderr:
            handle = win32.STDERR
        win32.SetConsoleTextAttribute(handle, attrs)

    def get_position(self, handle):
        position = win32.GetConsoleScreenBufferInfo(handle).dwCursorPosition
        position.X += 1
        position.Y += 1
        return position

    def set_cursor_position(self, position=None, on_stderr=False):
        if position is None:
            return
        handle = win32.STDOUT
        if on_stderr:
            handle = win32.STDERR
        win32.SetConsoleCursorPosition(handle, position)

    def cursor_up(self, num_rows=0, on_stderr=False):
        if num_rows == 0:
            return
        handle = win32.STDOUT
        if on_stderr:
            handle = win32.STDERR
        position = self.get_position(handle)
        adjusted_position = (position.Y - num_rows, position.X)
        self.set_cursor_position(adjusted_position, on_stderr)

    def erase_data(self, mode=0, on_stderr=False):
        if mode[0] not in (2,):
            return
        handle = win32.STDOUT
        if on_stderr:
            handle = win32.STDERR
        coord_screen = win32.COORD(0, 0)
        csbi = win32.GetConsoleScreenBufferInfo(handle)
        dw_con_size = csbi.dwSize.X * csbi.dwSize.Y
        win32.FillConsoleOutputCharacter(handle, ' ', dw_con_size, coord_screen)
        win32.FillConsoleOutputAttribute(handle, self.get_attrs(), dw_con_size, coord_screen)
        win32.SetConsoleCursorPosition(handle, (coord_screen.X, coord_screen.Y))