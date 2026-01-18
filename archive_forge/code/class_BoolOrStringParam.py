from typing import Union
import click
import functools
class BoolOrStringParam(click.ParamType):
    """A click parameter that can be either a boolean or a string."""
    name = 'BOOL | TEXT'

    def convert(self, value, param, ctx):
        if isinstance(value, bool):
            return value
        else:
            return bool_cast(value)