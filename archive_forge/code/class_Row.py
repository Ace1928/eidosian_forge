from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run.integrations import deployment_states
from googlecloudsdk.command_lib.run.integrations.formatters import base
class Row(object):
    """Represents the fields that will be used in the output of the table.

  Having a single class that has the expected values here is better than passing
  around a dict as the keys could mispelled or changed in one place.
  """

    def __init__(self, integration_name, integration_type, services, latest_deployment_status, region: str):
        self.integration_name = integration_name
        self.integration_type = integration_type
        self.services = services
        self.latest_deployment_status = latest_deployment_status
        self.region = region
        self.formatted_latest_deployment_status = _GetSymbolFromDeploymentStatus(latest_deployment_status)

    def __eq__(self, other):
        return self.integration_name == other.integration_name and self.integration_type == other.integration_type and (self.services == other.services) and (self.latest_deployment_status == other.latest_deployment_status) and (self.region == other.region)