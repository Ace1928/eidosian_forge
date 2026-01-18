import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
class FauxExtension(object):

    def __init__(self, *args, **kwds):
        self.args = args
        self.kwds = kwds

    def get_args_and_data(self, data):
        return (self.args, self.kwds, data)