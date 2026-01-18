import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
class VisitorContext:
    """Stores a list of QObject-derived classes encountered in order to find
       out which classes inherit QObject."""

    def __init__(self):
        self.qobject_derived = QOBJECT_DERIVED