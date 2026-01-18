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
def _inputs_help(cls):
    """
    Prints description for input parameters

    >>> from nipype.interfaces.afni import GCOR
    >>> _inputs_help(GCOR)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    ['Inputs::', '', '\\t[Mandatory]', '\\tin_file: (a pathlike object or string...

    """
    helpstr = ['Inputs::']
    mandatory_keys = []
    optional_items = []
    if cls.input_spec:
        inputs = cls.input_spec()
        mandatory_items = list(inputs.traits(mandatory=True).items())
        if mandatory_items:
            helpstr += ['', '\t[Mandatory]']
            for name, spec in mandatory_items:
                helpstr += get_trait_desc(inputs, name, spec)
        mandatory_keys = {item[0] for item in mandatory_items}
        optional_items = ['\n'.join(get_trait_desc(inputs, name, val)) for name, val in inputs.traits(transient=None).items() if name not in mandatory_keys]
        if optional_items:
            helpstr += ['', '\t[Optional]'] + optional_items
    if not mandatory_keys and (not optional_items):
        helpstr += ['', '\tNone']
    return helpstr