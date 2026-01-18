import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _get_matching_blocks(old, new):
    matcher = patiencediff.PatienceSequenceMatcher(None, old, new)
    return matcher.get_matching_blocks()