import copy
import importlib
from kivy.logger import Logger
from kivy.context import register_context
import kivy.factory_registers  # NOQA
class FactoryException(Exception):
    pass