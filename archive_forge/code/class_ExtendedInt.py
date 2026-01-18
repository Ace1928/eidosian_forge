import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class ExtendedInt(wsme.types.UserType):
    basetype = int
    name = 'Extended integer'