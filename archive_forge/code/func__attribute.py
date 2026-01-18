import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _attribute(node: ast.Attribute) -> Tuple[str, str]:
    """Split an attribute."""
    return (node.value.id, node.attr)