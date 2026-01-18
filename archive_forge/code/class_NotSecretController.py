import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
class NotSecretController(object):

    @expose()
    def _lookup(self, someID, *remainder):
        if someID == 'notfound':
            return None
        return (SubController(someID), remainder)
    unlocked = unlocked(SubController('unlocked'))