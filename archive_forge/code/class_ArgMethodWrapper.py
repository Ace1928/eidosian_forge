import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
class ArgMethodWrapper(ArgDecorator):
    """
    Base class to define a wrapper for ArgumentParser method.

    Child class must define either `_method_name` or `add_to_parser`.

    """
    _method_name: str

    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

    def add_to_parser(self, parser, group):
        """ Add this object's information to the parser.
        """
        if group is not None:
            parser = group
        getattr(parser, self._method_name)(*self.args, **self.kwds)
        return None