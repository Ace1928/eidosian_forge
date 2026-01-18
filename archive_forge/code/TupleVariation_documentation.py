from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
Return 2 if coordinates are (x, y) as in gvar, 1 if single values
        as in cvar, or 0 if empty.
        