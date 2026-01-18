import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
class TestObjectDispatch(PecanTestCase):

    @property
    def app_(self):

        class SubSubController(object):

            @expose()
            def index(self):
                return '/sub/sub/'

            @expose()
            def deeper(self):
                return '/sub/sub/deeper'

        class SubController(object):

            @expose()
            def index(self):
                return '/sub/'

            @expose()
            def deeper(self):
                return '/sub/deeper'
            sub = SubSubController()

        class RootController(object):

            @expose()
            def index(self):
                return '/'

            @expose()
            def deeper(self):
                return '/deeper'
            sub = SubController()
        return TestApp(Pecan(RootController()))

    def test_index(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert r.body == b'/'

    def test_one_level(self):
        r = self.app_.get('/deeper')
        assert r.status_int == 200
        assert r.body == b'/deeper'

    def test_one_level_with_trailing(self):
        r = self.app_.get('/sub/')
        assert r.status_int == 200
        assert r.body == b'/sub/'

    def test_two_levels(self):
        r = self.app_.get('/sub/deeper')
        assert r.status_int == 200
        assert r.body == b'/sub/deeper'

    def test_two_levels_with_trailing(self):
        r = self.app_.get('/sub/sub/')
        assert r.status_int == 200

    def test_three_levels(self):
        r = self.app_.get('/sub/sub/deeper')
        assert r.status_int == 200
        assert r.body == b'/sub/sub/deeper'