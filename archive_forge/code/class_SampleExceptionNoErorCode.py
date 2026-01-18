from heat.common import exception
from heat.tests import common
class SampleExceptionNoErorCode(exception.HeatException):
    msg_fmt = 'Test exception'