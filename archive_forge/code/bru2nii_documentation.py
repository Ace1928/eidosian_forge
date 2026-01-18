import os
from .base import (
Uses bru2nii's Bru2 to convert Bruker files

    Examples
    ========

    >>> from nipype.interfaces.bru2nii import Bru2
    >>> converter = Bru2()
    >>> converter.inputs.input_dir = "brukerdir"
    >>> converter.cmdline  # doctest: +ELLIPSIS
    'Bru2 -o .../data/brukerdir brukerdir'
    