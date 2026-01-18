import os
import warnings
from datetime import date
from inspect import cleandoc
from textwrap import indent
from typing import Optional, Tuple
class InformationOnly(SetuptoolsWarning):
    """Currently there is no clear way of displaying messages to the users
    that use the setuptools backend directly via ``pip``.
    The only thing that might work is a warning, although it is not the
    most appropriate tool for the job...

    See pypa/packaging-problems#558.
    """