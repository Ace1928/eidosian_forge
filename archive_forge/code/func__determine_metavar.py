from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _determine_metavar(self, action: argparse.Action, default_metavar: Union[str, Tuple[str, ...]]) -> Union[str, Tuple[str, ...]]:
    """Custom method to determine what to use as the metavar value of an action"""
    if action.metavar is not None:
        result = action.metavar
    elif action.choices is not None:
        choice_strs = [str(choice) for choice in action.choices]
        result = '{%s}' % ', '.join(choice_strs)
    else:
        result = default_metavar
    return result