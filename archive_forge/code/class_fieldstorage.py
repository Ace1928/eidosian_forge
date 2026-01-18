import re
import unittest
from wsme import exc
from wsme import types
class fieldstorage:
    filename = 'static.json'
    file = None
    type = 'application/json'
    value = 'from-value'