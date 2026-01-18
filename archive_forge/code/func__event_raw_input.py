from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
@Win32EventHandler(WM_INPUT)
def _event_raw_input(self, msg, wParam, lParam):
    hRawInput = cast(lParam, HRAWINPUT)
    inp = RAWINPUT()
    size = UINT(sizeof(inp))
    _user32.GetRawInputData(hRawInput, RID_INPUT, byref(inp), byref(size), sizeof(RAWINPUTHEADER))
    if inp.header.dwType == RIM_TYPEMOUSE:
        if not self._exclusive_mouse:
            return 0
        rmouse = inp.data.mouse
        if rmouse.usFlags & 1 == MOUSE_MOVE_RELATIVE:
            if rmouse.lLastX != 0 or rmouse.lLastY != 0:
                if self._exclusive_mouse_buttons:
                    self.dispatch_event('on_mouse_drag', 0, 0, rmouse.lLastX, -rmouse.lLastY, self._exclusive_mouse_buttons, self._get_modifiers())
                else:
                    self.dispatch_event('on_mouse_motion', 0, 0, rmouse.lLastX, -rmouse.lLastY)
        else:
            if self._exclusive_mouse_lpos is None:
                self._exclusive_mouse_lpos = (rmouse.lLastX, rmouse.lLastY)
            last_x, last_y = self._exclusive_mouse_lpos
            rel_x = rmouse.lLastX - last_x
            rel_y = rmouse.lLastY - last_y
            if rel_x != 0 or rel_y != 0.0:
                if self._exclusive_mouse_buttons:
                    self.dispatch_event('on_mouse_drag', 0, 0, rmouse.lLastX, -rmouse.lLastY, self._exclusive_mouse_buttons, self._get_modifiers())
                else:
                    self.dispatch_event('on_mouse_motion', 0, 0, rel_x, rel_y)
                self._exclusive_mouse_lpos = (rmouse.lLastX, rmouse.lLastY)
    elif inp.header.dwType == RIM_TYPEKEYBOARD:
        if inp.data.keyboard.VKey == 255:
            return 0
        key_up = inp.data.keyboard.Flags & RI_KEY_BREAK
        if inp.data.keyboard.MakeCode == 42:
            if not key_up and (not self._keyboard_state[42]):
                self._keyboard_state[42] = True
                self.dispatch_event('on_key_press', key.LSHIFT, self._get_modifiers())
            elif key_up and self._keyboard_state[42]:
                self._keyboard_state[42] = False
                self.dispatch_event('on_key_release', key.LSHIFT, self._get_modifiers())
        elif inp.data.keyboard.MakeCode == 54:
            if not key_up and (not self._keyboard_state[54]):
                self._keyboard_state[54] = True
                self.dispatch_event('on_key_press', key.RSHIFT, self._get_modifiers())
            elif key_up and self._keyboard_state[54]:
                self._keyboard_state[54] = False
                self.dispatch_event('on_key_release', key.RSHIFT, self._get_modifiers())
    return 0