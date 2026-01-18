from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
@runtime_checkable
class ChoicesProviderFuncBase(Protocol):
    """
    Function that returns a list of choices in support of tab completion
    """

    def __call__(self) -> List[str]:
        ...