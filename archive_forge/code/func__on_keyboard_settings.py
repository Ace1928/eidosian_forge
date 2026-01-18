from appearing. If you want to prevent the settings instance from appearing
from the same thread and the other coroutines are only executed when Kivy
import os
from inspect import getfile
from os.path import dirname, join, exists, sep, expanduser, isfile
from kivy.config import ConfigParser
from kivy.base import runTouchApp, async_runTouchApp, stopTouchApp
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty
from kivy.setupconfig import USE_SDL2
def _on_keyboard_settings(self, window, *largs):
    key = largs[0]
    setting_key = 282
    if platform == 'android' and (not USE_SDL2):
        import pygame
        setting_key = pygame.K_MENU
    if key == setting_key:
        if not self.open_settings():
            self.close_settings()
        return True
    if key == 27:
        return self.close_settings()