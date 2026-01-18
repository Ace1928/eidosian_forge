import sys
import runpy  # noqa: E402
from importlib.machinery import PathFinder  # noqa: E402
from os.path import dirname  # noqa: E402
Execute exactly this copy of pip, within a different environment.

This file is named as it is, to ensure that this module can't be imported via
an import statement.
