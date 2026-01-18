import collections
from oslo_serialization import jsonutils
from heat.api.aws import utils as aws_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
class GetAZs(function.Function):
    """A function for retrieving the availability zones.

    Takes the form::

        { "Fn::GetAZs" : "<region>" }
    """

    def result(self):
        if self.stack is None:
            return ['nova']
        else:
            return self.stack.get_availability_zones()