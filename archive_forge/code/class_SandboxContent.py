from functools import wraps
from kivy.context import Context
from kivy.base import ExceptionManagerBase
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
class SandboxContent(RelativeLayout):
    pass