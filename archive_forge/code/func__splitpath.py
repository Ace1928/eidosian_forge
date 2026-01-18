import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
def _splitpath(self, path):
    components = []
    head, tail = os.path.split(os.path.normpath(path))
    while head != path:
        if tail:
            components.append(tail)
        path = head
        head, tail = os.path.split(path)
    if head:
        components.append(head)
    components.reverse()
    return components