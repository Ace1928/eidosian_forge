import pygame
from kivy.compat import PY2
from kivy.core.window import WindowBase
from kivy.core import CoreCriticalException
from os import environ
from os.path import exists, join
from kivy.config import Config
from kivy import kivy_data_dir
from kivy.base import ExceptionManager
from kivy.logger import Logger
from kivy.base import stopTouchApp, EventLoop
from kivy.utils import platform, deprecated
from kivy.resources import resource_find
def _set_icon_win(self, filename):
    if not filename.endswith('.ico'):
        filename = '{}.ico'.format(filename.rsplit('.', 1)[0])
    if not exists(filename):
        return False
    import win32api
    import win32gui
    import win32con
    hwnd = pygame.display.get_wm_info()['window']
    icon_big = win32gui.LoadImage(None, filename, win32con.IMAGE_ICON, 48, 48, win32con.LR_LOADFROMFILE)
    icon_small = win32gui.LoadImage(None, filename, win32con.IMAGE_ICON, 16, 16, win32con.LR_LOADFROMFILE)
    win32api.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, icon_small)
    win32api.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, icon_big)
    return True