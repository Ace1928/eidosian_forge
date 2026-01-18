from __future__ import unicode_literals
from itertools import tee, chain
import re
import copy
class JsonPointerException(Exception):
    pass