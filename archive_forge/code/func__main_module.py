import os
import sys
from setuptools.command import easy_install
from os_ken import version
def _main_module():
    return sys.modules['__main__']