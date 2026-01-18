import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
class SecureController(object):

    @expose(generic=True)
    def index(self):
        return 'I should not be allowed'

    @index.when(method='POST')
    def index_post(self):
        return 'I should not be allowed'

    @expose(generic=True)
    def secret(self):
        return 'I should not be allowed'