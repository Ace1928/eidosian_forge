from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def _update_modifiers(self, mods=None, key=None):
    if mods is None and key is None:
        return
    modifiers = set()
    if mods is not None:
        if mods & (KMOD_RSHIFT | KMOD_LSHIFT):
            modifiers.add('shift')
        if mods & (KMOD_RALT | KMOD_LALT | KMOD_MODE):
            modifiers.add('alt')
        if mods & (KMOD_RCTRL | KMOD_LCTRL):
            modifiers.add('ctrl')
        if mods & (KMOD_RGUI | KMOD_LGUI):
            modifiers.add('meta')
        if mods & KMOD_NUM:
            modifiers.add('numlock')
        if mods & KMOD_CAPS:
            modifiers.add('capslock')
    if key is not None:
        if key in (KMOD_RSHIFT, KMOD_LSHIFT):
            modifiers.add('shift')
        if key in (KMOD_RALT, KMOD_LALT, KMOD_MODE):
            modifiers.add('alt')
        if key in (KMOD_RCTRL, KMOD_LCTRL):
            modifiers.add('ctrl')
        if key in (KMOD_RGUI, KMOD_LGUI):
            modifiers.add('meta')
        if key == KMOD_NUM:
            modifiers.add('numlock')
        if key == KMOD_CAPS:
            modifiers.add('capslock')
    self._modifiers = list(modifiers)
    return