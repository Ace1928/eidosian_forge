import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
@_registered_volume_type
class _InMemoryVolume(_VolumeType):
    """Volume Type representing an in-memory emptydir."""

    @classmethod
    def name(cls):
        return 'in-memory'

    @classmethod
    def help(cls):
        return "An ephemeral volume that stores data in the instance's memory. With this type of volume, data is not shared between instances and all data will be lost when the instance it is on is terminated."

    @classmethod
    def required_fields(cls):
        return {}

    @classmethod
    def optional_fields(cls):
        return {'size-limit': 'A quantity representing the maximum amount of memory allocated to this volume, such as "512Mi" or "3G". Data stored in an in-memory volume consumes the memory allocation of the container that wrote the data. If size-limit is not specified, the maximum size will be half the total memory limit of all containers.'}

    @classmethod
    def fill_volume(cls, volume, new_vol, messages):
        if 'size-limit' in volume:
            src = messages.EmptyDirVolumeSource(medium='Memory', sizeLimit=volume['size-limit'])
        else:
            src = messages.EmptyDirVolumeSource(medium='Memory')
        new_vol.emptyDir = src