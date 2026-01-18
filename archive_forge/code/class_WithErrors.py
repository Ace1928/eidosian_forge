import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class WithErrors(object):

    @expose()
    def divide_by_zero(self):
        1 / 0