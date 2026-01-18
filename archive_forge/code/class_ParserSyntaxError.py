from typing import Dict, Type
from parso import tree
from parso.pgen2.generator import ReservedString
class ParserSyntaxError(Exception):
    """
    Contains error information about the parser tree.

    May be raised as an exception.
    """

    def __init__(self, message, error_leaf):
        self.message = message
        self.error_leaf = error_leaf