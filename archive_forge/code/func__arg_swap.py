import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def _arg_swap(args: Union[Sequence[Any]], search_arg: Any, *replace_arg: Any) -> List[Any]:
    """
    Helper function for cmd2 decorators to swap the Statement parameter with one or more decorator-specific parameters

    :param args: The original positional arguments
    :param search_arg: The argument to search for (usually the Statement)
    :param replace_arg: The arguments to substitute in
    :return: The new set of arguments to pass to the command function
    """
    index = args.index(search_arg)
    args_list = list(args)
    args_list[index:index + 1] = replace_arg
    return args_list