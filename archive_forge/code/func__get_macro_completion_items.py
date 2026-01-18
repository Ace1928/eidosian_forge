import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _get_macro_completion_items(self) -> List[CompletionItem]:
    """Return list of macro names and values as CompletionItems"""
    results: List[CompletionItem] = []
    for cur_key in self.macros:
        row_data = [self.macros[cur_key].value]
        results.append(CompletionItem(cur_key, self._macro_completion_table.generate_data_row(row_data)))
    return results