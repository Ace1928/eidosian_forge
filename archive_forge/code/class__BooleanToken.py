import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
class _BooleanToken(_Token):
    kind = 'boolean'

    def value(self):
        return self.raw == 'true'