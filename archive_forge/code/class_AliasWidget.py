import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class AliasWidget(Label):
    _button = None

    def setter(self, value):
        self._button = value
        return True

    def getter(self):
        return self._button
    button = AliasProperty(getter, setter, rebind=True)