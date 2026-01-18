from __future__ import annotations
import enum
import typing
from ctypes import POINTER, Structure, Union, windll
from ctypes.wintypes import BOOL, CHAR, DWORD, HANDLE, LPDWORD, SHORT, UINT, WCHAR, WORD
class uChar(Union):
    """https://docs.microsoft.com/en-us/windows/console/key-event-record-str"""
    _fields_: typing.ClassVar[list[tuple[str, type]]] = [('AsciiChar', CHAR), ('UnicodeChar', WCHAR)]