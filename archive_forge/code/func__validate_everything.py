import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def _validate_everything(item):
    """ Item validator which accepts any item and returns it unaltered.
    """
    return item