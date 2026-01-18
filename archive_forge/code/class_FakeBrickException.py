from os_brick import exception
from os_brick.tests import base
class FakeBrickException(exception.BrickException):
    code = 500