from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
class ModuleContext:
    """Context of a module

    You can access to the config with self.config.
    """

    def __init__(self):
        self.config = {}

    def __repr__(self):
        return repr(self.config)