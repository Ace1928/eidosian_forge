import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
class Autoreloader(Monitor):
    """Monitor which re-executes the process when files change.

    This :ref:`plugin<plugins>` restarts the process (via :func:`os.execv`)
    if any of the files it monitors change (or is deleted). By default, the
    autoreloader monitors all imported modules; you can add to the
    set by adding to ``autoreload.files``::

        cherrypy.engine.autoreload.files.add(myFile)

    If there are imported files you do *not* wish to monitor, you can
    adjust the ``match`` attribute, a regular expression. For example,
    to stop monitoring cherrypy itself::

        cherrypy.engine.autoreload.match = r'^(?!cherrypy).+'

    Like all :class:`Monitor<cherrypy.process.plugins.Monitor>` plugins,
    the autoreload plugin takes a ``frequency`` argument. The default is
    1 second; that is, the autoreloader will examine files once each second.
    """
    files = None
    'The set of files to poll for modifications.'
    frequency = 1
    'The interval in seconds at which to poll for modified files.'
    match = '.*'
    'A regular expression by which to match filenames.'

    def __init__(self, bus, frequency=1, match='.*'):
        self.mtimes = {}
        self.files = set()
        self.match = match
        Monitor.__init__(self, bus, self.run, frequency)

    def start(self):
        """Start our own background task thread for self.run."""
        if self.thread is None:
            self.mtimes = {}
        Monitor.start(self)
    start.priority = 70

    def sysfiles(self):
        """Return a Set of sys.modules filenames to monitor."""
        search_mod_names = filter(re.compile(self.match).match, list(sys.modules.keys()))
        mods = map(sys.modules.get, search_mod_names)
        return set(filter(None, map(self._file_for_module, mods)))

    @classmethod
    def _file_for_module(cls, module):
        """Return the relevant file for the module."""
        return cls._archive_for_zip_module(module) or cls._file_for_file_module(module)

    @staticmethod
    def _archive_for_zip_module(module):
        """Return the archive filename for the module if relevant."""
        try:
            return module.__loader__.archive
        except AttributeError:
            pass

    @classmethod
    def _file_for_file_module(cls, module):
        """Return the file for the module."""
        try:
            return module.__file__ and cls._make_absolute(module.__file__)
        except AttributeError:
            pass

    @staticmethod
    def _make_absolute(filename):
        """Ensure filename is absolute to avoid effect of os.chdir."""
        return filename if os.path.isabs(filename) else os.path.normpath(os.path.join(_module__file__base, filename))

    def run(self):
        """Reload the process if registered files have been modified."""
        for filename in self.sysfiles() | self.files:
            if filename:
                if filename.endswith('.pyc'):
                    filename = filename[:-1]
                oldtime = self.mtimes.get(filename, 0)
                if oldtime is None:
                    continue
                try:
                    mtime = os.stat(filename).st_mtime
                except OSError:
                    mtime = None
                if filename not in self.mtimes:
                    self.mtimes[filename] = mtime
                elif mtime is None or mtime > oldtime:
                    self.bus.log('Restarting because %s changed.' % filename)
                    self.thread.cancel()
                    self.bus.log('Stopped thread %r.' % self.thread.name)
                    self.bus.restart()
                    return