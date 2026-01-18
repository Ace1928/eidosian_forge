import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class ObjWidgetRebindFalse(Label):
    button = ObjectProperty(None, rebind=False, allownone=True)