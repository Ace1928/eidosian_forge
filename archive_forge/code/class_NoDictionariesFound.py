import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
class NoDictionariesFound(Exception):
    """
    There aren't any dictionaries installed on the current system so
    spellchecking could not work in any way.
    """