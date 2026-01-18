from __future__ import absolute_import
from boto.mturk.test.support import unittest
class SeleniumFailed(object):

    def __init__(self, message):
        self.message = message

    def __nonzero__(self):
        return False