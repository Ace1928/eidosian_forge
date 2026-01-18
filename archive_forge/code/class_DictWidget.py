import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class DictWidget(Label):
    button = DictProperty({'button': None}, rebind=True, allownone=True)