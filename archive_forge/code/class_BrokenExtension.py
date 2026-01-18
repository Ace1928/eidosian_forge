import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
class BrokenExtension(object):

    def __init__(self, *args, **kwds):
        raise IOError('Did not create')