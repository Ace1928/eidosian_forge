from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
class FakeSAObject(object):

    def __init__(self):
        self._sa_class_manager = object()
        self._sa_instance_state = 'awesome'
        self.id = 1
        self.first_name = 'Jonathan'
        self.last_name = 'LaCour'