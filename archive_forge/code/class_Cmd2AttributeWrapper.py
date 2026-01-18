from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
class Cmd2AttributeWrapper:
    """
    Wraps a cmd2-specific attribute added to an argparse Namespace.
    This makes it easy to know which attributes in a Namespace are
    arguments from a parser and which were added by cmd2.
    """

    def __init__(self, attribute: Any) -> None:
        self.__attribute = attribute

    def get(self) -> Any:
        """Get the value of the attribute"""
        return self.__attribute

    def set(self, new_val: Any) -> None:
        """Set the value of the attribute"""
        self.__attribute = new_val