from __future__ import absolute_import
from . import GrocLexer
from . import GrocParser
import antlr3
class GrocException(Exception):
    """An error occurred while parsing the groc input string."""