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
class TestAbort(PecanTestCase):

    def test_abort(self):

        class RootController(object):

            @expose()
            def index(self):
                abort(404)
        app = TestApp(Pecan(RootController()))
        r = app.get('/', status=404)
        assert r.status_int == 404

    def test_abort_with_detail(self):

        class RootController(object):

            @expose()
            def index(self):
                abort(status_code=401, detail='Not Authorized')
        app = TestApp(Pecan(RootController()))
        r = app.get('/', status=401)
        assert r.status_int == 401

    def test_abort_keeps_traceback(self):
        last_exc, last_traceback = (None, None)
        try:
            try:
                raise Exception('Bottom Exception')
            except:
                abort(404)
        except Exception:
            last_exc, _, last_traceback = sys.exc_info()
        assert last_exc is HTTPNotFound
        assert 'Bottom Exception' in traceback.format_tb(last_traceback)[-1]