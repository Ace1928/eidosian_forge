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
def _run_prepare(self):
    if not self.built:
        self.load_config()
        self.load_kv(filename=self.kv_file)
        root = self.build()
        if root:
            self.root = root
    if self.root:
        if not isinstance(self.root, Widget):
            Logger.critical('App.root must be an _instance_ of Widget')
            raise Exception('Invalid instance in App.root')
        from kivy.core.window import Window
        Window.add_widget(self.root)
    from kivy.base import EventLoop
    window = EventLoop.window
    if window:
        self._app_window = window
        window.set_title(self.get_application_name())
        icon = self.get_application_icon()
        if icon:
            window.set_icon(icon)
        self._install_settings_keys(window)
    else:
        Logger.critical('Application: No window is created. Terminating application run.')
        return
    self.dispatch('on_start')