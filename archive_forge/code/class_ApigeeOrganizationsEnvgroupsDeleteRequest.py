from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvgroupsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvgroupsDeleteRequest object.

  Fields:
    name: Required. Name of the environment group in the following format:
      `organizations/{org}/envgroups/{envgroup}`.
  """
    name = _messages.StringField(1, required=True)