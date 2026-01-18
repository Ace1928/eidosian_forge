from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_set_choices_provider(self: argparse.Action, choices_provider: ChoicesProviderFunc) -> None:
    """
    Set choices_provider of an argparse Action.

    This function is added by cmd2 as a method called ``set_choices_callable()`` to ``argparse.Action`` class.

    To call: ``action.set_choices_provider(choices_provider)``

    :param self: action being edited
    :param choices_provider: the choices_provider instance to use
    :raises: TypeError if used on incompatible action type
    """
    self._set_choices_callable(ChoicesCallable(is_completer=False, to_call=choices_provider))