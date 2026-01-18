import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
class ListedWindow:
    """
    Mixin class that allows emulation of the Apple Window menu. Windows
    register when they open by calling the class method register_window.  They
    call unregister_window when they close.  The class maintains a list of all
    openwindows.Participating windows should be subclasses of ListedWindow, as
    should objects which need access to its list of all windows in the app, such
    as the Settings object.
    """
    window_list = []
    settings = {}

    @classmethod
    def register_window(cls, window):
        assert isinstance(window, ListedWindow)
        cls.window_list.append(window)
        window.apply_settings()

    @classmethod
    def unregister_window(cls, window):
        try:
            cls.window_list.remove(window)
        except ValueError:
            pass

    def bring_to_front(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def apply_settings(self):
        pass