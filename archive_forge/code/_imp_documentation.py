from the deprecated imp module.
import os
import importlib.util
import importlib.machinery
from importlib.util import module_from_spec
Just like 'imp.find_module()', but with package support