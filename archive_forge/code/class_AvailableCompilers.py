from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re
from semantic_version import Version
@dataclasses.dataclass
class AvailableCompilers:
    """This contains data of installed PennyLane compiler packages."""
    entrypoints_interface = ('context', 'qjit', 'ops')
    names_entrypoints = defaultdict(dict)