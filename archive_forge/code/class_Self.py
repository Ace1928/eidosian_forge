import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
class Self(object):
    """ Singleton 'Self' object (used as object reference to current 'object').
    """

    def __repr__(self):
        return '<self>'