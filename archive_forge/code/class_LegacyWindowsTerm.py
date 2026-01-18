import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
class LegacyWindowsTerm:
    """This class allows interaction with the legacy Windows Console API. It should only be used in the context
    of environments where virtual terminal processing is not available. However, if it is used in a Windows environment,
    the entire API should work.

    Args:
        file (IO[str]): The file which the Windows Console API HANDLE is retrieved from, defaults to sys.stdout.
    """
    BRIGHT_BIT = 8
    ANSI_TO_WINDOWS = [0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15]

    def __init__(self, file: 'IO[str]') -> None:
        handle = GetStdHandle(STDOUT)
        self._handle = handle
        default_text = GetConsoleScreenBufferInfo(handle).wAttributes
        self._default_text = default_text
        self._default_fore = default_text & 7
        self._default_back = default_text >> 4 & 7
        self._default_attrs = self._default_fore | self._default_back << 4
        self._file = file
        self.write = file.write
        self.flush = file.flush

    @property
    def cursor_position(self) -> WindowsCoordinates:
        """Returns the current position of the cursor (0-based)

        Returns:
            WindowsCoordinates: The current cursor position.
        """
        coord: COORD = GetConsoleScreenBufferInfo(self._handle).dwCursorPosition
        return WindowsCoordinates(row=cast(int, coord.Y), col=cast(int, coord.X))

    @property
    def screen_size(self) -> WindowsCoordinates:
        """Returns the current size of the console screen buffer, in character columns and rows

        Returns:
            WindowsCoordinates: The width and height of the screen as WindowsCoordinates.
        """
        screen_size: COORD = GetConsoleScreenBufferInfo(self._handle).dwSize
        return WindowsCoordinates(row=cast(int, screen_size.Y), col=cast(int, screen_size.X))

    def write_text(self, text: str) -> None:
        """Write text directly to the terminal without any modification of styles

        Args:
            text (str): The text to write to the console
        """
        self.write(text)
        self.flush()

    def write_styled(self, text: str, style: Style) -> None:
        """Write styled text to the terminal.

        Args:
            text (str): The text to write
            style (Style): The style of the text
        """
        color = style.color
        bgcolor = style.bgcolor
        if style.reverse:
            color, bgcolor = (bgcolor, color)
        if color:
            fore = color.downgrade(ColorSystem.WINDOWS).number
            fore = fore if fore is not None else 7
            if style.bold:
                fore = fore | self.BRIGHT_BIT
            if style.dim:
                fore = fore & ~self.BRIGHT_BIT
            fore = self.ANSI_TO_WINDOWS[fore]
        else:
            fore = self._default_fore
        if bgcolor:
            back = bgcolor.downgrade(ColorSystem.WINDOWS).number
            back = back if back is not None else 0
            back = self.ANSI_TO_WINDOWS[back]
        else:
            back = self._default_back
        assert fore is not None
        assert back is not None
        SetConsoleTextAttribute(self._handle, attributes=ctypes.c_ushort(fore | back << 4))
        self.write_text(text)
        SetConsoleTextAttribute(self._handle, attributes=self._default_text)

    def move_cursor_to(self, new_position: WindowsCoordinates) -> None:
        """Set the position of the cursor

        Args:
            new_position (WindowsCoordinates): The WindowsCoordinates representing the new position of the cursor.
        """
        if new_position.col < 0 or new_position.row < 0:
            return
        SetConsoleCursorPosition(self._handle, coords=new_position)

    def erase_line(self) -> None:
        """Erase all content on the line the cursor is currently located at"""
        screen_size = self.screen_size
        cursor_position = self.cursor_position
        cells_to_erase = screen_size.col
        start_coordinates = WindowsCoordinates(row=cursor_position.row, col=0)
        FillConsoleOutputCharacter(self._handle, ' ', length=cells_to_erase, start=start_coordinates)
        FillConsoleOutputAttribute(self._handle, self._default_attrs, length=cells_to_erase, start=start_coordinates)

    def erase_end_of_line(self) -> None:
        """Erase all content from the cursor position to the end of that line"""
        cursor_position = self.cursor_position
        cells_to_erase = self.screen_size.col - cursor_position.col
        FillConsoleOutputCharacter(self._handle, ' ', length=cells_to_erase, start=cursor_position)
        FillConsoleOutputAttribute(self._handle, self._default_attrs, length=cells_to_erase, start=cursor_position)

    def erase_start_of_line(self) -> None:
        """Erase all content from the cursor position to the start of that line"""
        row, col = self.cursor_position
        start = WindowsCoordinates(row, 0)
        FillConsoleOutputCharacter(self._handle, ' ', length=col, start=start)
        FillConsoleOutputAttribute(self._handle, self._default_attrs, length=col, start=start)

    def move_cursor_up(self) -> None:
        """Move the cursor up a single cell"""
        cursor_position = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=cursor_position.row - 1, col=cursor_position.col))

    def move_cursor_down(self) -> None:
        """Move the cursor down a single cell"""
        cursor_position = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=cursor_position.row + 1, col=cursor_position.col))

    def move_cursor_forward(self) -> None:
        """Move the cursor forward a single cell. Wrap to the next line if required."""
        row, col = self.cursor_position
        if col == self.screen_size.col - 1:
            row += 1
            col = 0
        else:
            col += 1
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=row, col=col))

    def move_cursor_to_column(self, column: int) -> None:
        """Move cursor to the column specified by the zero-based column index, staying on the same row

        Args:
            column (int): The zero-based column index to move the cursor to.
        """
        row, _ = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row, column))

    def move_cursor_backward(self) -> None:
        """Move the cursor backward a single cell. Wrap to the previous line if required."""
        row, col = self.cursor_position
        if col == 0:
            row -= 1
            col = self.screen_size.col - 1
        else:
            col -= 1
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row=row, col=col))

    def hide_cursor(self) -> None:
        """Hide the cursor"""
        current_cursor_size = self._get_cursor_size()
        invisible_cursor = CONSOLE_CURSOR_INFO(dwSize=current_cursor_size, bVisible=0)
        SetConsoleCursorInfo(self._handle, cursor_info=invisible_cursor)

    def show_cursor(self) -> None:
        """Show the cursor"""
        current_cursor_size = self._get_cursor_size()
        visible_cursor = CONSOLE_CURSOR_INFO(dwSize=current_cursor_size, bVisible=1)
        SetConsoleCursorInfo(self._handle, cursor_info=visible_cursor)

    def set_title(self, title: str) -> None:
        """Set the title of the terminal window

        Args:
            title (str): The new title of the console window
        """
        assert len(title) < 255, 'Console title must be less than 255 characters'
        SetConsoleTitle(title)

    def _get_cursor_size(self) -> int:
        """Get the percentage of the character cell that is filled by the cursor"""
        cursor_info = CONSOLE_CURSOR_INFO()
        GetConsoleCursorInfo(self._handle, cursor_info=cursor_info)
        return int(cursor_info.dwSize)