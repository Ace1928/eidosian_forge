import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (

            Check if an argument belongs to a mutually exclusive group and either mark that group
            as complete or print an error if the group has already been completed
            :param arg_action: the action of the argument
            :raises: CompletionError if the group is already completed
            