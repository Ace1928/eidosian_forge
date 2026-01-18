from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
@attr.s
class HTMLElement(object):
    """Holds an HTML element, as created by elementMaker."""
    name = attr.ib()
    children = attr.ib()
    attributes = attr.ib()