import os
import sys
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as importlib_entry_points
from typing import Iterable
from ._check_module import ArgcompleteMarkerNotFound, find

Utility for locating the module (or package's __init__.py)
associated with a given console_script name
and verifying it contains the PYTHON_ARGCOMPLETE_OK marker.

Such scripts are automatically generated and cannot contain
the marker themselves, so we defer to the containing module or package.

For more information on setuptools console_scripts, see
https://setuptools.readthedocs.io/en/latest/setuptools.html#automatic-script-creation

Intended to be invoked by argcomplete's global completion function.
