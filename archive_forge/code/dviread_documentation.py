from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook

        Parse a line in the font mapping file.

        The format is (partially) documented at
        http://mirrors.ctan.org/systems/doc/pdftex/manual/pdftex-a.pdf
        https://tug.org/texinfohtml/dvips.html#psfonts_002emap
        Each line can have the following fields:

        - tfmname (first, only required field),
        - psname (defaults to tfmname, must come immediately after tfmname if
          present),
        - fontflags (integer, must come immediately after psname if present,
          ignored by us),
        - special (SlantFont and ExtendFont, only field that is double-quoted),
        - fontfile, encodingfile (optional, prefixed by <, <<, or <[; << always
          precedes a font, <[ always precedes an encoding, < can precede either
          but then an encoding file must have extension .enc; < and << also
          request different font subsetting behaviors but we ignore that; < can
          be separated from the filename by whitespace).

        special, fontfile, and encodingfile can appear in any order.
        