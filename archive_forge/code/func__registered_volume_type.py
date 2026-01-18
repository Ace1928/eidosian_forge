import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
def _registered_volume_type(cls):
    """decorator for registering VolumeTypes.

  Only VolumeTypes with this decorator will be supported in add_volume

  Args:
    cls: the decorated class

  Returns:
    cls
  """
    _supported_volume_types[cls.name()] = cls
    return cls