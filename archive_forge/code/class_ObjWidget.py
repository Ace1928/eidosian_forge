import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
class ObjWidget(Label):
    button = ObjectProperty(None, rebind=True, allownone=True)