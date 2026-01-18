import pythran.config as cfg
from collections import defaultdict
import os.path
import os

    Description of a Pythran extension

    Similar to distutils.core.Extension except that the sources are .py files
    They must be processable by pythran, of course.

    The compilation process ends up in a native Python module.
    