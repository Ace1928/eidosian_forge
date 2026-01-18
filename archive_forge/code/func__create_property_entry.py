import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _create_property_entry(self, name: str, type: str, getter: Optional[str]=None) -> PropertyEntry:
    """Create a property JSON entry."""
    result: PropertyEntry = {'name': name, 'type': type, 'index': len(self._properties)}
    if getter:
        result['read'] = getter
    return result