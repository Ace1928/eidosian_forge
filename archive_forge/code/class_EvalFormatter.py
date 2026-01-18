import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
class EvalFormatter(Formatter):
    """A String Formatter that allows evaluation of simple expressions.

    Note that this version interprets a `:`  as specifying a format string (as per
    standard string formatting), so if slicing is required, you must explicitly
    create a slice.

    This is to be used in templating cases, such as the parallel batch
    script templates, where simple arithmetic on arguments is useful.

    Examples
    --------
    ::

        In [1]: f = EvalFormatter()
        In [2]: f.format('{n//4}', n=8)
        Out[2]: '2'

        In [3]: f.format("{greeting[slice(2,4)]}", greeting="Hello")
        Out[3]: 'll'
    """

    def get_field(self, name, args, kwargs):
        v = eval(name, kwargs)
        return (v, name)