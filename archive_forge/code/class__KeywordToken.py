import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
class _KeywordToken(_Token):
    kind = 'keyword'

    def is_keyword(self, *names):
        return self.raw in names