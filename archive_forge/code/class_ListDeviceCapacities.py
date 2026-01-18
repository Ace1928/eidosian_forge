from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.firebase.test.device_capacities import DEVICE_CAPACITY_TABLE_FORMAT
from googlecloudsdk.api_lib.firebase.test.device_capacities import DeviceCapacities
from googlecloudsdk.calliope import base
class ListDeviceCapacities(base.ListCommand, DeviceCapacities):
    """List capacity information for iOS models & versions.

    List device capacity information (high/medium/low/none) for all iOS
    models & versions which are available for testing and have capacity
    information published.

    Device capacity is the number of online devices in Firebase Test Lab. For
    physical devices, the number is the average of online devices in the last 30
    days. It's important to note that device capacity does not directly reflect
    any real-time data, like the length of the test queue, or the
    available/busy state of the devices based on current test traffic.
  """

    @staticmethod
    def Args(parser):
        """Method called by Calliope to register flags for this command.

    Args:
      parser: An argparse parser used to add arguments that follow this command
        in the CLI. Positional arguments are allowed.
    """
        parser.display_info.AddFormat(DEVICE_CAPACITY_TABLE_FORMAT)
        base.URI_FLAG.RemoveFromParser(parser)

    def Run(self, args):
        """Run the 'gcloud firebase test ios list-device-capacities' command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation (i.e. group and command arguments combined).

    Returns:
      The list of device models, versions, and capacity info we want to have
      printed later. Obsolete (unsupported) devices, versions, and entries
      missing capacity info are filtered out.
    """
        return self.get_capacity_data(util.GetIosCatalog(self.context))