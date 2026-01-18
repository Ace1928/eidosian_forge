from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _ArgumentParser_get_ap_completer_type(self: argparse.ArgumentParser) -> Optional[Type['ArgparseCompleter']]:
    """
    Get the ap_completer_type attribute of an argparse ArgumentParser.

    This function is added by cmd2 as a method called ``get_ap_completer_type()`` to ``argparse.ArgumentParser`` class.

    To call: ``parser.get_ap_completer_type()``

    :param self: ArgumentParser being queried
    :return: An ArgparseCompleter-based class or None if attribute does not exist
    """
    return cast(Optional[Type['ArgparseCompleter']], getattr(self, ATTR_AP_COMPLETER_TYPE, None))