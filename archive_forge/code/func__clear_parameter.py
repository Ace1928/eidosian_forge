import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
def _clear_parameter(self, name):
    """Reset or clear a commandline option value (PRIVATE)."""
    cleared_option = False
    for parameter in self.parameters:
        if name in parameter.names:
            parameter.value = None
            parameter.is_set = False
            cleared_option = True
    if not cleared_option:
        raise ValueError(f'Option name {name} was not found.')