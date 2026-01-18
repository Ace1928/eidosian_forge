import pickle
import hashlib
import sys
import types
import struct
import io
import decimal
class _MyHash(object):
    """ Class used to hash objects that won't normally pickle """

    def __init__(self, *args):
        self.args = args