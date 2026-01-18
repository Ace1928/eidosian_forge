from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
class StrSubclass(str):

    def __getitem__(self, index):
        return StrSubclass(super().__getitem__(index))