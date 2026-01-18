from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnoseClusterRequest(_messages.Message):
    """A request to collect cluster diagnostic information.

  Enums:
    TarballAccessValueValuesEnum: Optional. (Optional) The access type to the
      diagnostic tarball. If not specified, falls back to default access of
      the bucket

  Fields:
    diagnosisInterval: Optional. Time interval in which diagnosis should be
      carried out on the cluster.
    job: Optional. DEPRECATED Specifies the job on which diagnosis is to be
      performed. Format: projects/{project}/regions/{region}/jobs/{job}
    jobs: Optional. Specifies a list of jobs on which diagnosis is to be
      performed. Format: projects/{project}/regions/{region}/jobs/{job}
    tarballAccess: Optional. (Optional) The access type to the diagnostic
      tarball. If not specified, falls back to default access of the bucket
    tarballGcsDir: Optional. (Optional) The output Cloud Storage directory for
      the diagnostic tarball. If not specified, a task-specific directory in
      the cluster's staging bucket will be used.
    workers: Optional. A list of workers in the cluster to run the diagnostic
      script on.
    yarnApplicationId: Optional. DEPRECATED Specifies the yarn application on
      which diagnosis is to be performed.
    yarnApplicationIds: Optional. Specifies a list of yarn applications on
      which diagnosis is to be performed.
  """

    class TarballAccessValueValuesEnum(_messages.Enum):
        """Optional. (Optional) The access type to the diagnostic tarball. If not
    specified, falls back to default access of the bucket

    Values:
      TARBALL_ACCESS_UNSPECIFIED: Tarball Access unspecified. Falls back to
        default access of the bucket
      GOOGLE_CLOUD_SUPPORT: Google Cloud Support group has read access to the
        diagnostic tarball
      GOOGLE_DATAPROC_DIAGNOSE: Google Cloud Dataproc Diagnose service account
        has read access to the diagnostic tarball
    """
        TARBALL_ACCESS_UNSPECIFIED = 0
        GOOGLE_CLOUD_SUPPORT = 1
        GOOGLE_DATAPROC_DIAGNOSE = 2
    diagnosisInterval = _messages.MessageField('Interval', 1)
    job = _messages.StringField(2)
    jobs = _messages.StringField(3, repeated=True)
    tarballAccess = _messages.EnumField('TarballAccessValueValuesEnum', 4)
    tarballGcsDir = _messages.StringField(5)
    workers = _messages.StringField(6, repeated=True)
    yarnApplicationId = _messages.StringField(7)
    yarnApplicationIds = _messages.StringField(8, repeated=True)