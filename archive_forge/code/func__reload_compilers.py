from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re
from semantic_version import Version
def _reload_compilers():
    """Reload and scan installed PennyLane compiler packages to refresh the
    compilers names and entry points.
    """
    reload(metadata)
    _refresh_compilers()