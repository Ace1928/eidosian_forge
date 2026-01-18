from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2ExfilResource(_messages.Message):
    """Resource where data was exfiltrated from or exfiltrated to.

  Fields:
    components: Subcomponents of the asset that was exfiltrated, like URIs
      used during exfiltration, table names, databases, and filenames. For
      example, multiple tables might have been exfiltrated from the same Cloud
      SQL instance, or multiple files might have been exfiltrated from the
      same Cloud Storage bucket.
    name: The resource's [full resource name](https://cloud.google.com/apis/de
      sign/resource_names#full_resource_name).
  """
    components = _messages.StringField(1, repeated=True)
    name = _messages.StringField(2)