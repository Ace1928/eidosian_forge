import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _index_of_property(self, name: str) -> int:
    """Search a property by name"""
    for i in range(len(self._properties)):
        if self._properties[i]['name'] == name:
            return i
    return -1