import weakref
from unittest import mock
from cliff import show
from cliff.tests import base
class FauxFormatter(object):

    def __init__(self):
        self.args = []
        self.obj = weakref.proxy(self)

    def emit_one(self, columns, data, stdout, args):
        self.args.append((columns, data))