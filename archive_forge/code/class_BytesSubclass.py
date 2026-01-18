from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
class BytesSubclass(bytes):

    def __getitem__(self, index):
        return BytesSubclass(super().__getitem__(index))