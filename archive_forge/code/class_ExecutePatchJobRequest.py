from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExecutePatchJobRequest(_messages.Message):
    """A request message to initiate patching across Compute Engine instances.

  Fields:
    description: Description of the patch job. Length of the description is
      limited to 1024 characters.
    displayName: Display name for this patch job. This does not have to be
      unique.
    dryRun: If this patch is a dry-run only, instances are contacted but will
      do nothing.
    duration: Duration of the patch job. After the duration ends, the patch
      job times out.
    instanceFilter: Required. Instances to patch, either explicitly or
      filtered by some criteria such as zone or labels.
    patchConfig: Patch configuration being applied. If omitted, instances are
      patched using the default configurations.
    rollout: Rollout strategy of the patch job.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    dryRun = _messages.BooleanField(3)
    duration = _messages.StringField(4)
    instanceFilter = _messages.MessageField('PatchInstanceFilter', 5)
    patchConfig = _messages.MessageField('PatchConfig', 6)
    rollout = _messages.MessageField('PatchRollout', 7)