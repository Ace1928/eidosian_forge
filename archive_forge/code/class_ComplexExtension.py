from __future__ import absolute_import, division, print_function
import bz2
import hashlib
import logging
import os
import re
import struct
import sys
import types
import zlib
from io import BytesIO
class ComplexExtension(Extension):
    name = 'c'
    cls = complex

    def encode(self, s, v):
        return (v.real, v.imag)

    def decode(self, s, v):
        return complex(v[0], v[1])