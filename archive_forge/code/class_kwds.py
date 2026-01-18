import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
class kwds(ArgDecorator):
    """ Provide other keywords to the sub-parser constructor.
    """

    def __init__(self, **kwds):
        self.kwds = kwds

    def __call__(self, func):
        func = super(kwds, self).__call__(func)
        func.argcmd_kwds = self.kwds
        return func