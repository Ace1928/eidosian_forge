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
def _pygame_set_mode(self, size=None):
    if size is None:
        size = self.size
    if self.fullscreen == 'auto':
        pygame.display.set_mode((0, 0), self.flags)
    else:
        pygame.display.set_mode(size, self.flags)