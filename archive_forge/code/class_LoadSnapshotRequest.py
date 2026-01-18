from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadSnapshotRequest(_messages.Message):
    """Request to load a snapshot into a Cloud Composer environment.

  Fields:
    skipAirflowOverridesSetting: Whether or not to skip setting Airflow
      overrides when loading the environment's state.
    skipEnvironmentVariablesSetting: Whether or not to skip setting
      environment variables when loading the environment's state.
    skipGcsDataCopying: Whether or not to skip copying Cloud Storage data when
      loading the environment's state.
    skipPypiPackagesInstallation: Whether or not to skip installing Pypi
      packages when loading the environment's state.
    snapshotPath: A Cloud Storage path to a snapshot to load, e.g.: "gs://my-
      bucket/snapshots/project_location_environment_timestamp".
  """
    skipAirflowOverridesSetting = _messages.BooleanField(1)
    skipEnvironmentVariablesSetting = _messages.BooleanField(2)
    skipGcsDataCopying = _messages.BooleanField(3)
    skipPypiPackagesInstallation = _messages.BooleanField(4)
    snapshotPath = _messages.StringField(5)