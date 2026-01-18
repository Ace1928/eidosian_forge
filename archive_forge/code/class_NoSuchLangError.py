import sys
from kivy.core import core_select_lib
class NoSuchLangError(Exception):
    """
    Exception to be raised when a specific language could not be found.
    """
    pass