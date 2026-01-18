from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
def activate_module(self, name, win):
    """Activate a module on a window"""
    if name not in self.mods:
        Logger.warning('Modules: Module <%s> not found' % name)
        return
    mod = self.mods[name]
    if 'module' not in mod:
        self._configure_module(name)
    pymod = mod['module']
    if not mod['activated']:
        context = mod['context']
        msg = 'Modules: Start <{0}> with config {1}'.format(name, context)
        Logger.debug(msg)
        pymod.start(win, context)
        mod['activated'] = True