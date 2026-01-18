from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _action_get_custom_parameter(self: argparse.Action) -> Any:
    f'\n        Get the custom {param_name} attribute of an argparse Action.\n\n        This function is added by cmd2 as a method called ``{getter_name}()`` to ``argparse.Action`` class.\n\n        To call: ``action.{getter_name}()``\n\n        :param self: argparse Action being queried\n        :return: The value of {param_name} or None if attribute does not exist\n        '
    return getattr(self, attr_name, None)