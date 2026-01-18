import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
def _write_lines(self, lines):
    if not lines:
        return
    lines = NLCRE.split(lines)
    for line in lines[:-1]:
        self.write(line)
        self.write(self._NL)
    if lines[-1]:
        self.write(lines[-1])