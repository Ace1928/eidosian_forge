from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsCachesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsCachesDeleteRequest object.

  Fields:
    name: Required. Cache resource name of the form: `organizations/{organizat
      ion_id}/environments/{environment_id}/caches/{cache_id}`
  """
    name = _messages.StringField(1, required=True)