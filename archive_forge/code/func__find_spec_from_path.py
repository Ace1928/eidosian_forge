from ._abc import Loader
from ._bootstrap import module_from_spec
from ._bootstrap import _resolve_name
from ._bootstrap import spec_from_loader
from ._bootstrap import _find_spec
from ._bootstrap_external import MAGIC_NUMBER
from ._bootstrap_external import _RAW_MAGIC_NUMBER
from ._bootstrap_external import cache_from_source
from ._bootstrap_external import decode_source
from ._bootstrap_external import source_from_cache
from ._bootstrap_external import spec_from_file_location
from contextlib import contextmanager
import _imp
import functools
import sys
import types
import warnings
def _find_spec_from_path(name, path=None):
    """Return the spec for the specified module.

    First, sys.modules is checked to see if the module was already imported. If
    so, then sys.modules[name].__spec__ is returned. If that happens to be
    set to None, then ValueError is raised. If the module is not in
    sys.modules, then sys.meta_path is searched for a suitable spec with the
    value of 'path' given to the finders. None is returned if no spec could
    be found.

    Dotted names do not have their parent packages implicitly imported. You will
    most likely need to explicitly import all parent packages in the proper
    order for a submodule to get the correct spec.

    """
    if name not in sys.modules:
        return _find_spec(name, path)
    else:
        module = sys.modules[name]
        if module is None:
            return None
        try:
            spec = module.__spec__
        except AttributeError:
            raise ValueError('{}.__spec__ is not set'.format(name)) from None
        else:
            if spec is None:
                raise ValueError('{}.__spec__ is None'.format(name))
            return spec