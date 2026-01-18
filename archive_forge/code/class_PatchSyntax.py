import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class PatchSyntax(BzrError):
    """Base class for patch syntax errors."""