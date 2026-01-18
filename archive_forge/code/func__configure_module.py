from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
def _configure_module(self, name):
    if 'module' not in self.mods[name]:
        try:
            self.import_module(name)
        except ImportError:
            return
    config = dict()
    args = Config.get('modules', name)
    if args != '':
        values = Config.get('modules', name).split(',')
        for value in values:
            x = value.split('=', 1)
            if len(x) == 1:
                config[x[0]] = True
            else:
                config[x[0]] = x[1]
    self.mods[name]['context'].config = config
    if hasattr(self.mods[name]['module'], 'configure'):
        self.mods[name]['module'].configure(config)