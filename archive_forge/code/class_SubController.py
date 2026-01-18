from json import dumps
from webtest import TestApp
from pecan import Pecan, expose, abort
from pecan.tests import PecanTestCase
class SubController(object):
    sub = SubSubController()