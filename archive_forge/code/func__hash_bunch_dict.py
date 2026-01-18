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
def _hash_bunch_dict(adict, key):
    """Inject file hashes into adict[key]"""
    stuff = adict[key]
    if not is_container(stuff):
        stuff = [stuff]
    return [(afile, hash_infile(afile)) for afile in stuff]