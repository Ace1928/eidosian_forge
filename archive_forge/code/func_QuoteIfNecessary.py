import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def QuoteIfNecessary(string):
    """TODO: Should this ideally be replaced with one or more of the above
    functions?"""
    if '"' in string:
        string = '"' + string.replace('"', '\\"') + '"'
    return string