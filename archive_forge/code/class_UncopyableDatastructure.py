import logging
import os
import pathlib
import sys
import time
import pytest
class UncopyableDatastructure:

    def __init__(self, name):
        self._lock = threading.Lock()
        self._name = name

    def __str__(self):
        return 'UncopyableDatastructure(name=%r)' % self._name