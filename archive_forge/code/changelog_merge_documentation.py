import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
Merge changelog changes.

         * new entries from other will float to the top
         * edits to older entries are preserved
        