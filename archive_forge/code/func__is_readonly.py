import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
def _is_readonly(volume):
    return 'readonly' in volume and volume['readonly'].lower() == 'true'