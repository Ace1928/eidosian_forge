from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import datetime
import os
class InMemoryHistory(History):
    """
    :class:`.History` class that keeps a list of all strings in memory.
    """

    def __init__(self):
        self.strings = []

    def append(self, string):
        self.strings.append(string)

    def __getitem__(self, key):
        return self.strings[key]

    def __iter__(self):
        return iter(self.strings)

    def __len__(self):
        return len(self.strings)