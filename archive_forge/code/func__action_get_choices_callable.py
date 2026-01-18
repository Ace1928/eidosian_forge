from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_get_choices_callable(self: argparse.Action) -> Optional[ChoicesCallable]:
    """
    Get the choices_callable attribute of an argparse Action.

    This function is added by cmd2 as a method called ``get_choices_callable()`` to ``argparse.Action`` class.

    To call: ``action.get_choices_callable()``

    :param self: argparse Action being queried
    :return: A ChoicesCallable instance or None if attribute does not exist
    """
    return cast(Optional[ChoicesCallable], getattr(self, ATTR_CHOICES_CALLABLE, None))