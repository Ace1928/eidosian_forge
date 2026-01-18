import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
class SuffixImporter(object):
    """An importer designed using the mechanism defined in :pep:`302`. I read
    the PEP, and also used Doug Hellmann's PyMOTW article `Modules and
    Imports`_, as a pattern.

    .. _`Modules and Imports`: http://www.doughellmann.com/PyMOTW/sys/imports.html

    Define a subclass that specifies a :attr:`suffix` attribute, and
    implements a :meth:`process_filedata` method. Then call the classmethod
    :meth:`register` on your class to actually install it in the appropriate
    places in :mod:`sys`. """
    scheme = 'suffix'
    suffix = None
    path_entry = None

    @classmethod
    def trigger_url(cls):
        if cls.suffix is None:
            raise ValueError('%s.suffix is not set' % cls.__name__)
        return 'suffix:%s' % cls.suffix

    @classmethod
    def register(cls):
        sys.path_hooks.append(cls)
        sys.path.append(cls.trigger_url())

    def __init__(self, path_entry):
        pr = url_parse(str(path_entry))
        if pr.scheme != self.scheme or pr.path != self.suffix:
            raise ImportError()
        self.path_entry = path_entry
        self._found = {}

    def checkpath_iter(self, fullname):
        for dirpath in sys.path:
            finder = sys.path_importer_cache.get(dirpath)
            if isinstance(finder, (type(None), importlib.machinery.FileFinder)):
                checkpath = os.path.join(dirpath, '{0}.{1}'.format(fullname, self.suffix))
                yield checkpath

    def find_module(self, fullname, path=None):
        for checkpath in self.checkpath_iter(fullname):
            if os.path.isfile(checkpath):
                self._found[fullname] = checkpath
                return self
        return None

    def load_module(self, fullname):
        assert fullname in self._found
        if fullname in sys.modules:
            module = sys.modules[fullname]
        else:
            sys.modules[fullname] = module = types.ModuleType(fullname)
        data = None
        with open(self._found[fullname]) as f:
            data = f.read()
        module.__dict__.clear()
        module.__file__ = self._found[fullname]
        module.__name__ = fullname
        module.__loader__ = self
        self.process_filedata(module, data)
        return module

    def process_filedata(self, module, data):
        pass