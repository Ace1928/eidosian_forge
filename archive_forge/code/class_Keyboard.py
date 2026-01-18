from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
class Keyboard(EventDispatcher):
    """Keyboard interface that is returned by
    :meth:`WindowBase.request_keyboard`. When you request a keyboard,
    you'll get an instance of this class. Whatever the keyboard input is
    (system or virtual keyboard), you'll receive events through this
    instance.

    :Events:
        `on_key_down`: keycode, text, modifiers
            Fired when a new key is pressed down
        `on_key_up`: keycode
            Fired when a key is released (up)

    Here is an example of how to request a Keyboard in accordance with the
    current configuration:

    .. include:: ../../examples/widgets/keyboardlistener.py
        :literal:

    """
    keycodes = {'backspace': 8, 'tab': 9, 'enter': 13, 'rshift': 303, 'shift': 304, 'alt': 308, 'rctrl': 306, 'lctrl': 305, 'super': 309, 'alt-gr': 307, 'compose': 311, 'pipe': 310, 'capslock': 301, 'escape': 27, 'spacebar': 32, 'pageup': 280, 'pagedown': 281, 'end': 279, 'home': 278, 'left': 276, 'up': 273, 'right': 275, 'down': 274, 'insert': 277, 'delete': 127, 'numlock': 300, 'print': 144, 'screenlock': 145, 'pause': 19, 'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101, 'f': 102, 'g': 103, 'h': 104, 'i': 105, 'j': 106, 'k': 107, 'l': 108, 'm': 109, 'n': 110, 'o': 111, 'p': 112, 'q': 113, 'r': 114, 's': 115, 't': 116, 'u': 117, 'v': 118, 'w': 119, 'x': 120, 'y': 121, 'z': 122, '0': 48, '1': 49, '2': 50, '3': 51, '4': 52, '5': 53, '6': 54, '7': 55, '8': 56, '9': 57, 'numpad0': 256, 'numpad1': 257, 'numpad2': 258, 'numpad3': 259, 'numpad4': 260, 'numpad5': 261, 'numpad6': 262, 'numpad7': 263, 'numpad8': 264, 'numpad9': 265, 'numpaddecimal': 266, 'numpaddivide': 267, 'numpadmul': 268, 'numpadsubstract': 269, 'numpadadd': 270, 'numpadenter': 271, 'f1': 282, 'f2': 283, 'f3': 284, 'f4': 285, 'f5': 286, 'f6': 287, 'f7': 288, 'f8': 289, 'f9': 290, 'f10': 291, 'f11': 292, 'f12': 293, 'f13': 294, 'f14': 295, 'f15': 296, '(': 40, ')': 41, '[': 91, ']': 93, '{': 123, '}': 125, ':': 58, ';': 59, '=': 61, '+': 43, '-': 45, '_': 95, '/': 47, '*': 42, '?': 47, '`': 96, '~': 126, '´': 180, '¦': 166, '\\': 92, '|': 124, '"': 34, "'": 39, ',': 44, '.': 46, '<': 60, '>': 62, '@': 64, '!': 33, '#': 35, '$': 36, '%': 37, '^': 94, '&': 38, '¬': 172, '¨': 168, '…': 8230, 'ù': 249, 'à': 224, 'é': 233, 'è': 232}
    __events__ = ('on_key_down', 'on_key_up', 'on_textinput')

    def __init__(self, **kwargs):
        super(Keyboard, self).__init__()
        self.window = kwargs.get('window', None)
        self.callback = kwargs.get('callback', None)
        self.target = kwargs.get('target', None)
        self.widget = kwargs.get('widget', None)

    def on_key_down(self, keycode, text, modifiers):
        pass

    def on_key_up(self, keycode):
        pass

    def on_textinput(self, text):
        pass

    def release(self):
        """Call this method to release the current keyboard.
        This will ensure that the keyboard is no longer attached to your
        callback."""
        if self.window:
            self.window.release_keyboard(self.target)
            self.target = None

    def _on_window_textinput(self, instance, text):
        return self.dispatch('on_textinput', text)

    def _on_window_key_down(self, instance, keycode, scancode, text, modifiers):
        keycode = (keycode, self.keycode_to_string(keycode))
        if text == '\x04':
            Window.trigger_keyboard_height()
            return
        return self.dispatch('on_key_down', keycode, text, modifiers)

    def _on_window_key_up(self, instance, keycode, *largs):
        keycode = (keycode, self.keycode_to_string(keycode))
        return self.dispatch('on_key_up', keycode)

    def _on_vkeyboard_key_down(self, instance, keycode, text, modifiers):
        if keycode is None:
            keycode = text.lower()
        keycode = (self.string_to_keycode(keycode), keycode)
        return self.dispatch('on_key_down', keycode, text, modifiers)

    def _on_vkeyboard_key_up(self, instance, keycode, text, modifiers):
        if keycode is None:
            keycode = text
        keycode = (self.string_to_keycode(keycode), keycode)
        return self.dispatch('on_key_up', keycode)

    def _on_vkeyboard_textinput(self, instance, text):
        return self.dispatch('on_textinput', text)

    def string_to_keycode(self, value):
        """Convert a string to a keycode number according to the
        :attr:`Keyboard.keycodes`. If the value is not found in the
        keycodes, it will return -1.
        """
        return Keyboard.keycodes.get(value, -1)

    def keycode_to_string(self, value):
        """Convert a keycode number to a string according to the
        :attr:`Keyboard.keycodes`. If the value is not found in the
        keycodes, it will return ''.
        """
        keycodes = list(Keyboard.keycodes.values())
        if value in keycodes:
            return list(Keyboard.keycodes.keys())[keycodes.index(value)]
        return ''