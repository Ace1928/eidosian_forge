import argparse
import sys
from typing import Iterable, List, Optional
from .. import __version__
from ..universaldetector import UniversalDetector

    Handles command line arguments and gets things started.

    :param argv: List of arguments, as if specified on the command-line.
                 If None, ``sys.argv[1:]`` is used instead.
    :type argv: list of str
    