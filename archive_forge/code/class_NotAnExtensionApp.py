import importlib
import time
import warnings
class NotAnExtensionApp(Exception):
    """An error raised when a module is not an extension."""