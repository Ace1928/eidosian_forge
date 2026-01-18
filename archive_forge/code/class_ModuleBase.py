from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
class ModuleBase:
    """Handle Kivy modules. It will automatically load and instantiate the
    module for the general window."""

    def __init__(self, **kwargs):
        self.mods = {}
        self.wins = []

    def add_path(self, path):
        """Add a path to search for modules in"""
        if not os.path.exists(path):
            return
        if path not in sys.path:
            sys.path.append(path)
        dirs = os.listdir(path)
        for module in dirs:
            name, ext = os.path.splitext(module)
            if ext not in ('.py', '.pyo', '.pyc') or name == '__init__':
                continue
            self.mods[name] = {'name': name, 'activated': False, 'context': ModuleContext()}

    def list(self):
        """Return the list of available modules"""
        return self.mods

    def import_module(self, name):
        try:
            modname = 'kivy.modules.{0}'.format(name)
            module = importlib.__import__(name=modname)
            module = sys.modules[modname]
        except ImportError:
            try:
                module = importlib.__import__(name=name)
                module = sys.modules[name]
            except ImportError:
                Logger.exception('Modules: unable to import <%s>' % name)
                self.mods[name]['module'] = None
                return
        if not hasattr(module, 'start'):
            Logger.warning('Modules: Module <%s> missing start() function' % name)
            return
        if not hasattr(module, 'stop'):
            err = 'Modules: Module <%s> missing stop() function' % name
            Logger.warning(err)
            return
        self.mods[name]['module'] = module

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

    def deactivate_module(self, name, win):
        """Deactivate a module from a window"""
        if name not in self.mods:
            Logger.warning('Modules: Module <%s> not found' % name)
            return
        if 'module' not in self.mods[name]:
            return
        module = self.mods[name]['module']
        if self.mods[name]['activated']:
            module.stop(win, self.mods[name]['context'])
            self.mods[name]['activated'] = False

    def register_window(self, win):
        """Add the window to the window list"""
        if win not in self.wins:
            self.wins.append(win)
        self.update()

    def unregister_window(self, win):
        """Remove the window from the window list"""
        if win in self.wins:
            self.wins.remove(win)
        self.update()

    def update(self):
        """Update the status of the module for each window"""
        modules_to_activate = [x[0] for x in Config.items('modules')]
        for win in self.wins:
            for name in self.mods:
                if name not in modules_to_activate:
                    self.deactivate_module(name, win)
            for name in modules_to_activate:
                try:
                    self.activate_module(name, win)
                except:
                    import traceback
                    traceback.print_exc()
                    raise

    def configure(self):
        """(internal) Configure all the modules before using them.
        """
        modules_to_configure = [x[0] for x in Config.items('modules')]
        for name in modules_to_configure:
            if name not in self.mods:
                Logger.warning('Modules: Module <%s> not found' % name)
                continue
            self._configure_module(name)

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

    def usage_list(self):
        print('Available modules')
        print('=================')
        for module in sorted(self.list()):
            if 'module' not in self.mods[module]:
                self.import_module(module)
            if not self.mods[module]['module'].__doc__:
                continue
            text = self.mods[module]['module'].__doc__.strip('\n ')
            text = text.split('\n')
            if len(text) > 2:
                if text[1].startswith('='):
                    text[1] = '=' * (14 + len(text[1]))
            text = '\n'.join(text)
            print('\n%-12s: %s' % (module, text))