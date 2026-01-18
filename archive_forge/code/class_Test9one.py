from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
class Test9one(ResponseElement):
    Nest = Element()
    Zoom = Element()