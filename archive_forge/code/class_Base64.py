import collections
from oslo_serialization import jsonutils
from heat.api.aws import utils as aws_utils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
class Base64(function.Function):
    """A placeholder function for converting to base64.

    Takes the form::

        { "Fn::Base64" : "<string>" }

    This function actually performs no conversion. It is included for the
    benefit of templates that convert UserData to Base64. Heat accepts UserData
    in plain text.
    """

    def result(self):
        resolved = function.resolve(self.args)
        if not isinstance(resolved, str):
            raise TypeError(_('"%s" argument must be a string') % self.fn_name)
        return resolved