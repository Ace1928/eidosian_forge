import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
class defaults(ArgMethodWrapper):
    """ Store arguments and keywords to pass to set_defaults().

    Instances also serve to decorate command methods.
    """
    _method_name = 'set_defaults'