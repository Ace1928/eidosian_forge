from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ArgListWithRequiredFieldsCheck(arg_parsers.ArgList):
    """ArgList that raises errror if required fields are not present."""

    def __call__(self, arg_value):
        arglist = super(ArgListWithRequiredFieldsCheck, self).__call__(arg_value)
        missing_required_fields = set(REQUIRED_INVENTORY_REPORTS_METADATA_FIELDS) - set(arglist)
        if missing_required_fields:
            raise arg_parsers.ArgumentTypeError('Fields {} are REQUIRED.'.format(','.join(sorted(missing_required_fields))))
        return arglist