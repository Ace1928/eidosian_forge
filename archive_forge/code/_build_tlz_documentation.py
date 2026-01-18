import sys
import types
import toolz
from importlib import import_module
from importlib.machinery import ModuleSpec
 Finds and loads ``tlz`` modules when added to sys.meta_path