from pytest import raises
import time
from promise import Promise, promisify, is_thenable
class MyType(object):
    pass