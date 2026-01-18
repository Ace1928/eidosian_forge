import sys
import os
from numpy.core import dtype
from numpy.core import numerictypes as _numerictypes
from numpy.core.function_base import add_newdoc

This file is separate from ``_add_newdocs.py`` so that it can be mocked out by
our sphinx ``conf.py`` during doc builds, where we want to avoid showing
platform-dependent information.
