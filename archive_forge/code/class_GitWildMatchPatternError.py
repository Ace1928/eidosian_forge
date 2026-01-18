from Rsync's wildmatch. Git uses wildmatch for its ".gitignore" files.
import re
import warnings
from typing import (
from .. import util
from ..pattern import RegexPattern
class GitWildMatchPatternError(ValueError):
    """
	The :class:`GitWildMatchPatternError` indicates an invalid git wild match
	pattern.
	"""
    pass