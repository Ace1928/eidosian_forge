from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MergeVersionAliasesRequest(_messages.Message):
    """Request message for ModelService.MergeVersionAliases.

  Fields:
    versionAliases: Required. The set of version aliases to merge. The alias
      should be at most 128 characters, and match `a-z{0,126}[a-z-0-9]`. Add
      the `-` prefix to an alias means removing that alias from the version.
      `-` is NOT counted in the 128 characters. Example: `-golden` means
      removing the `golden` alias from the version. There is NO ordering in
      aliases, which means 1) The aliases returned from GetModel API might not
      have the exactly same order from this MergeVersionAliases API. 2) Adding
      and deleting the same alias in the request is not recommended, and the 2
      operations will be cancelled out.
  """
    versionAliases = _messages.StringField(1, repeated=True)