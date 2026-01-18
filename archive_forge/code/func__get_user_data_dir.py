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
def _get_user_data_dir(self):
    data_dir = ''
    if platform == 'ios':
        data_dir = expanduser(join('~/Documents', self.name))
    elif platform == 'android':
        from jnius import autoclass, cast
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        context = cast('android.content.Context', PythonActivity.mActivity)
        file_p = cast('java.io.File', context.getFilesDir())
        data_dir = file_p.getAbsolutePath()
    elif platform == 'win':
        data_dir = os.path.join(os.environ['APPDATA'], self.name)
    elif platform == 'macosx':
        data_dir = '~/Library/Application Support/{}'.format(self.name)
        data_dir = expanduser(data_dir)
    else:
        data_dir = os.environ.get('XDG_CONFIG_HOME', '~/.config')
        data_dir = expanduser(join(data_dir, self.name))
    if not exists(data_dir):
        os.mkdir(data_dir)
    return data_dir