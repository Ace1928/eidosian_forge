import os
from contextlib import AbstractContextManager
from copy import deepcopy
from textwrap import wrap
import re
from datetime import datetime as dt
from dateutil.parser import parse as parseutc
import platform
from ... import logging, config
from ...utils.misc import is_container, rgetcwd
from ...utils.filemanip import md5, hash_infile
def _refs_help(cls):
    """Prints interface references."""
    references = getattr(cls, '_references', None)
    if not references:
        return []
    helpstr = ['References:', '-----------']
    for r in references:
        helpstr += ['{}'.format(r['entry'])]
    return helpstr