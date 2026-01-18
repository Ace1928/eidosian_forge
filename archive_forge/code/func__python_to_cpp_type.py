import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _python_to_cpp_type(type: str) -> str:
    """Python to C++ type"""
    c = CPP_TYPE_MAPPING.get(type)
    return c if c else type