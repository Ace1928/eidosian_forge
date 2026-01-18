import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
class magic_arguments(ArgDecorator):
    """ Mark the magic as having argparse arguments and possibly adjust the
    name.
    """

    def __init__(self, name=None):
        self.name = name

    def __call__(self, func):
        if not getattr(func, 'has_arguments', False):
            func.has_arguments = True
            func.decorators = []
        if self.name is not None:
            func.argcmd_name = self.name
        func.parser = construct_parser(func)
        return func