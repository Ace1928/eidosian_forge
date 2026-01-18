from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RuntimeTraceConfigOverride(_messages.Message):
    """NEXT ID: 7 Trace configuration override for a specific API proxy in an
  environment.

  Fields:
    apiProxy: Name of the API proxy that will have its trace configuration
      overridden following format: `organizations/{org}/apis/{api}`
    name: Name of the trace config override in the following format:
      `organizations/{org}/environment/{env}/traceConfig/overrides/{override}`
    revisionCreateTime: The timestamp that the revision was created or
      updated.
    revisionId: Revision number which can be used by the runtime to detect if
      the trace config override has changed between two versions.
    samplingConfig: Trace configuration override for a specific API proxy in
      an environment.
    uid: Unique ID for the configuration override. The ID will only change if
      the override is deleted and recreated. Corresponds to name's "override"
      field.
  """
    apiProxy = _messages.StringField(1)
    name = _messages.StringField(2)
    revisionCreateTime = _messages.StringField(3)
    revisionId = _messages.StringField(4)
    samplingConfig = _messages.MessageField('GoogleCloudApigeeV1RuntimeTraceSamplingConfig', 5)
    uid = _messages.StringField(6)