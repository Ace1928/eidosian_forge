from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_set_choices_callable(self: argparse.Action, choices_callable: ChoicesCallable) -> None:
    """
    Set the choices_callable attribute of an argparse Action.

    This function is added by cmd2 as a method called ``_set_choices_callable()`` to ``argparse.Action`` class.

    Call this using the convenience wrappers ``set_choices_provider()`` and ``set_completer()`` instead.

    :param self: action being edited
    :param choices_callable: the ChoicesCallable instance to use
    :raises: TypeError if used on incompatible action type
    """
    if self.choices is not None:
        err_msg = 'None of the following parameters can be used alongside a choices parameter:\nchoices_provider, completer'
        raise TypeError(err_msg)
    elif self.nargs == 0:
        err_msg = 'None of the following parameters can be used on an action that takes no arguments:\nchoices_provider, completer'
        raise TypeError(err_msg)
    setattr(self, ATTR_CHOICES_CALLABLE, choices_callable)