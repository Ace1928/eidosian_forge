from binascii import hexlify
from collections.abc import MutableMapping
from collections import OrderedDict
from enum import Enum
import itertools
from json import JSONEncoder
from warnings import warn
from fiona.errors import FionaDeprecationWarning
class _Geometry:

    def __init__(self, coordinates=None, type=None, geometries=None):
        self.coordinates = coordinates
        self.type = type
        self.geometries = geometries