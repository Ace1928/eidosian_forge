from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsResourcefilesListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsResourcefilesListRequest object.

  Fields:
    parent: Required. Name of the environment in which to list resource files
      in the following format: `organizations/{org}/environments/{env}`.
    type: Optional. Type of resource files to list. {{ resource_file_type }}
  """
    parent = _messages.StringField(1, required=True)
    type = _messages.StringField(2)