from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _ArgumentParser_set_ap_completer_type(self: argparse.ArgumentParser, ap_completer_type: Type['ArgparseCompleter']) -> None:
    """
    Set the ap_completer_type attribute of an argparse ArgumentParser.

    This function is added by cmd2 as a method called ``set_ap_completer_type()`` to ``argparse.ArgumentParser`` class.

    To call: ``parser.set_ap_completer_type(ap_completer_type)``

    :param self: ArgumentParser being edited
    :param ap_completer_type: the custom ArgparseCompleter-based class to use when tab completing arguments for this parser
    """
    setattr(self, ATTR_AP_COMPLETER_TYPE, ap_completer_type)