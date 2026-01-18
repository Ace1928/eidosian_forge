from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
class FakeResultProxy(ResultProxy):

    def __init__(self):
        self.rowcount = -1
        self.rows = []

    def __iter__(self):
        return iter(self.rows)

    def append(self, row):
        self.rows.append(row)