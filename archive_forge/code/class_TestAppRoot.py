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
class TestAppRoot(PecanTestCase):

    def test_controller_lookup_by_string_path(self):
        app = Pecan('pecan.tests.test_base.SampleRootController')
        assert app.root and app.root.__class__.__name__ == 'SampleRootController'