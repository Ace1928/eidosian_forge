from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NasJobSpec(_messages.Message):
    """Represents the spec of a NasJob.

  Fields:
    enableNasControllerMigration: Indicate if nas should launch controller
      migrated pipeline. This flag is for internal testing only, and will be
      removed once the migration is completed.
    multiTrialAlgorithmSpec: The spec of multi-trial algorithms.
    resumeNasJobId: The ID of the existing NasJob in the same Project and
      Location which will be used to resume search. search_space_spec and
      nas_algorithm_spec are obtained from previous NasJob hence should not
      provide them again for this NasJob.
    searchSpaceSpec: It defines the search space for Neural Architecture
      Search (NAS).
  """
    enableNasControllerMigration = _messages.BooleanField(1)
    multiTrialAlgorithmSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJobSpecMultiTrialAlgorithmSpec', 2)
    resumeNasJobId = _messages.StringField(3)
    searchSpaceSpec = _messages.StringField(4)