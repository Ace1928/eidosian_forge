import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@_registered_volume_type
class _GcsVolume(_VolumeType):
    """Volume Type representing a GCS volume."""

    @classmethod
    def name(cls):
        return 'cloud-storage'

    @classmethod
    def help(cls):
        return 'A volume representing a Cloud Storage bucket. This volume type is mounted using Cloud Storage FUSE. See https://cloud.google.com/storage/docs/gcs-fuse for the details and limitations of this filesystem.'

    @classmethod
    def required_fields(cls):
        return {'bucket': 'the name of the bucket to use as the source of this volume'}

    @classmethod
    def optional_fields(cls):
        return {'readonly': 'A boolean. If true, this volume will be read-only from all mounts.'}

    @classmethod
    def fill_volume(cls, volume, new_vol, messages):
        src = messages.CSIVolumeSource(driver='gcsfuse.run.googleapis.com', readOnly=_is_readonly(volume))
        src.volumeAttributes = messages.CSIVolumeSource.VolumeAttributesValue()
        src.volumeAttributes.additionalProperties.append(messages.CSIVolumeSource.VolumeAttributesValue.AdditionalProperty(key='bucketName', value=volume['bucket']))
        new_vol.csi = src