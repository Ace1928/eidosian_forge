from __future__ import annotations
import operator
import sys
import os
import re as _re
import struct
from textwrap import fill, dedent
class Undecidable(ValueError):
    pass