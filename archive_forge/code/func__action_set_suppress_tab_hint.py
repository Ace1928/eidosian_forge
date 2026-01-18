from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_set_suppress_tab_hint(self: argparse.Action, suppress_tab_hint: bool) -> None:
    """
    Set the suppress_tab_hint attribute of an argparse Action.

    This function is added by cmd2 as a method called ``set_suppress_tab_hint()`` to ``argparse.Action`` class.

    To call: ``action.set_suppress_tab_hint(suppress_tab_hint)``

    :param self: argparse Action being updated
    :param suppress_tab_hint: value being assigned
    """
    setattr(self, ATTR_SUPPRESS_TAB_HINT, suppress_tab_hint)