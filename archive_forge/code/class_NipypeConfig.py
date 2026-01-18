import os
import sys
import errno
import atexit
from warnings import warn
from looseversion import LooseVersion
import configparser
import numpy as np
from simplejson import load, dump
from .misc import str2bool
from filelock import SoftFileLock
class NipypeConfig(object):
    """Base nipype config class"""

    def __init__(self, *args, **kwargs):
        self._config = configparser.ConfigParser()
        self._cwd = None
        config_dir = os.path.expanduser(os.getenv('NIPYPE_CONFIG_DIR', default='~/.nipype'))
        self.data_file = os.path.join(config_dir, 'nipype.json')
        self.set_default_config()
        self._display = None
        self._resource_monitor = None
        self._config.read([os.path.join(config_dir, 'nipype.cfg'), 'nipype.cfg'])
        for option in CONFIG_DEPRECATIONS:
            for section in ['execution', 'logging', 'monitoring']:
                if self.has_option(section, option):
                    new_section, new_option = CONFIG_DEPRECATIONS[option][0].split('.')
                    if not self.has_option(new_section, new_option):
                        self.set(new_section, new_option, self.get(section, option))

    @property
    def cwd(self):
        """Cache current working directory ASAP"""
        if self._cwd is None:
            try:
                self._cwd = os.getcwd()
            except OSError:
                warn('Trying to run Nipype from a nonexistent directory "{}".'.format(os.getenv('PWD', 'unknown')), RuntimeWarning)
                raise
        return self._cwd

    def set_default_config(self):
        """Read default settings template and set into config object"""
        default_cfg = DEFAULT_CONFIG_TPL.format(log_dir=os.path.expanduser('~'), crashdump_dir=self.cwd)
        try:
            self._config.read_string(default_cfg)
        except AttributeError:
            from io import StringIO
            self._config.readfp(StringIO(default_cfg))

    def enable_debug_mode(self):
        """Enables debug configuration"""
        from .. import logging
        self._config.set('execution', 'stop_on_first_crash', 'true')
        self._config.set('execution', 'remove_unnecessary_outputs', 'false')
        self._config.set('execution', 'keep_inputs', 'true')
        self._config.set('logging', 'workflow_level', 'DEBUG')
        self._config.set('logging', 'interface_level', 'DEBUG')
        self._config.set('logging', 'utils_level', 'DEBUG')
        logging.update_logging(self._config)

    def set_log_dir(self, log_dir):
        """Sets logging directory

        This should be the first thing that is done before any nipype class
        with logging is imported.
        """
        self._config.set('logging', 'log_directory', log_dir)

    def get(self, section, option, default=None):
        """Get an option"""
        if option in CONFIG_DEPRECATIONS:
            msg = 'Config option "%s" has been deprecated as of nipype %s. Please use "%s" instead.' % (option, CONFIG_DEPRECATIONS[option][1], CONFIG_DEPRECATIONS[option][0])
            warn(msg)
            section, option = CONFIG_DEPRECATIONS[option][0].split('.')
        if self._config.has_option(section, option):
            return self._config.get(section, option)
        return default

    def set(self, section, option, value):
        """Set new value on option"""
        if isinstance(value, bool):
            value = str(value)
        if option in CONFIG_DEPRECATIONS:
            msg = 'Config option "%s" has been deprecated as of nipype %s. Please use "%s" instead.' % (option, CONFIG_DEPRECATIONS[option][1], CONFIG_DEPRECATIONS[option][0])
            warn(msg)
            section, option = CONFIG_DEPRECATIONS[option][0].split('.')
        return self._config.set(section, option, value)

    def getboolean(self, section, option):
        """Get a boolean option from section"""
        return self._config.getboolean(section, option)

    def has_option(self, section, option):
        """Check if option exists in section"""
        return self._config.has_option(section, option)

    @property
    def _sections(self):
        return self._config._sections

    def get_data(self, key):
        """Read options file"""
        if not os.path.exists(self.data_file):
            return None
        with SoftFileLock('%s.lock' % self.data_file):
            with open(self.data_file, 'rt') as file:
                datadict = load(file)
        if key in datadict:
            return datadict[key]
        return None

    def save_data(self, key, value):
        """Store config file"""
        datadict = {}
        if os.path.exists(self.data_file):
            with SoftFileLock('%s.lock' % self.data_file):
                with open(self.data_file, 'rt') as file:
                    datadict = load(file)
        else:
            dirname = os.path.dirname(self.data_file)
            if not os.path.exists(dirname):
                mkdir_p(dirname)
        with SoftFileLock('%s.lock' % self.data_file):
            with open(self.data_file, 'wt') as file:
                datadict[key] = value
                dump(datadict, file)

    def update_config(self, config_dict):
        """Extend internal dictionary with config_dict"""
        for section in ['execution', 'logging', 'check']:
            if section in config_dict:
                for key, val in list(config_dict[section].items()):
                    if not key.startswith('__'):
                        self._config.set(section, key, str(val))

    def update_matplotlib(self):
        """Set backend on matplotlib from options"""
        import matplotlib
        matplotlib.use(self.get('execution', 'matplotlib_backend'))

    def enable_provenance(self):
        """Sets provenance storing on"""
        self._config.set('execution', 'write_provenance', 'true')
        self._config.set('execution', 'hash_method', 'content')

    @property
    def resource_monitor(self):
        """Check if resource_monitor is available"""
        if self._resource_monitor is not None:
            return self._resource_monitor
        self.resource_monitor = str2bool(self._config.get('monitoring', 'enabled')) or False
        return self._resource_monitor

    @resource_monitor.setter
    def resource_monitor(self, value):
        if isinstance(value, (str, bytes)):
            value = str2bool(value.lower())
        if value is False:
            self._resource_monitor = False
        elif value is True:
            if not self._resource_monitor:
                self._resource_monitor = False
                try:
                    import psutil
                    self._resource_monitor = LooseVersion(psutil.__version__) >= LooseVersion('5.0')
                except ImportError:
                    pass
                finally:
                    if not self._resource_monitor:
                        warn('Could not enable the resource monitor: psutil>=5.0 could not be imported.')
                    self._config.set('monitoring', 'enabled', ('%s' % self._resource_monitor).lower())

    def enable_resource_monitor(self):
        """Sets the resource monitor on"""
        self.resource_monitor = True

    def disable_resource_monitor(self):
        """Sets the resource monitor off"""
        self.resource_monitor = False

    def get_display(self):
        """Returns the first display available"""
        if self._display is not None:
            return ':%d' % self._display.new_display
        sysdisplay = None
        if self._config.has_option('execution', 'display_variable'):
            sysdisplay = self._config.get('execution', 'display_variable')
        sysdisplay = sysdisplay or os.getenv('DISPLAY')
        if sysdisplay:
            from collections import namedtuple

            def _mock():
                pass
            ndisp = sysdisplay.split(':')[-1].split('.')[0]
            Xvfb = namedtuple('Xvfb', ['new_display', 'stop'])
            self._display = Xvfb(int(ndisp), _mock)
            return self.get_display()
        else:
            if 'darwin' in sys.platform:
                raise RuntimeError('Xvfb requires root permissions to run in OSX. Please make sure that an X server is listening and set the appropriate config on either $DISPLAY or nipype\'s "display_variable" config. Valid X servers include VNC, XQuartz, or manually started Xvfb.')
            if sysdisplay == '':
                del os.environ['DISPLAY']
            try:
                from xvfbwrapper import Xvfb
            except ImportError:
                raise RuntimeError('A display server was required, but $DISPLAY is not defined and Xvfb could not be imported.')
            self._display = Xvfb(nolisten='tcp')
            self._display.start()
            if not hasattr(self._display, 'new_display'):
                setattr(self._display, 'new_display', self._display.vdisplay_num)
            return self.get_display()

    def stop_display(self):
        """Closes the display if started"""
        if self._display is not None:
            from .. import logging
            self._display.stop()
            logging.getLogger('nipype.interface').debug('Closing display (if virtual)')