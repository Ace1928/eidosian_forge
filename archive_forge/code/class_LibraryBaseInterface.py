import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
class LibraryBaseInterface(BaseInterface):
    _pkg = None
    imports = ()

    def __init__(self, check_import=True, *args, **kwargs):
        super(LibraryBaseInterface, self).__init__(*args, **kwargs)
        if check_import:
            import pkgutil
            failed_imports = []
            for pkg in (self._pkg,) + tuple(self.imports):
                if pkgutil.find_loader(pkg) is None:
                    failed_imports.append(pkg)
            if failed_imports:
                iflogger.warning('Unable to import %s; %s interface may fail to run', failed_imports, self.__class__.__name__)

    @property
    def version(self):
        if self._version is None:
            import importlib
            try:
                self._version = importlib.import_module(self._pkg).__version__
            except (ImportError, AttributeError):
                pass
        return super(LibraryBaseInterface, self).version