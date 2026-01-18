import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
@staticmethod
def _count_changed_regions(old_lines, new_lines):
    matcher = patiencediff.PatienceSequenceMatcher(None, old_lines, new_lines)
    blocks = matcher.get_matching_blocks()
    return len(blocks) - 2