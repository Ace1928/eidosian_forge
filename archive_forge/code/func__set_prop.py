import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def _set_prop(self, value):
    self.setter_called += 1
    self.base_value = value / 2
    return True