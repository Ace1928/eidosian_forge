from typing import Any
from triad.exceptions import NoneArgumentError
Assert an argument is not None, otherwise raise exception

    :param obj: argument value
    :param arg_name: argument name, if None or empty, it will use `msg`
    :param msg: only when `arg_name` is None or empty, this value is used

    :raises NoneArgumentError: with `arg_name` or `msg`
    