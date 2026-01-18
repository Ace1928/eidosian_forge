from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
@runtime_checkable
class ChoicesProviderFuncWithTokens(Protocol):
    """
    Function that returns a list of choices in support of tab completion and accepts a dictionary of prior arguments.
    """

    def __call__(self, *, arg_tokens: Dict[str, List[str]]={}) -> List[str]:
        ...