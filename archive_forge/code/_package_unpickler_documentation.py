import _compat_pickle
import pickle
from .importer import Importer
Package-aware unpickler.

    This behaves the same as a normal unpickler, except it uses `importer` to
    find any global names that it encounters while unpickling.
    