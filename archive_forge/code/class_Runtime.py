from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.calliope import base
class Runtime:
    """Runtimes wrapper for ListRuntimesResponse#Runtimes.

  Attributes:
    name: A string name of the runtime.
    stage: An enum of the release state of the runtime, e.g., GA, BETA, etc.
    environment: Environment of the runtime.
  """

    def __init__(self, runtime):
        self.name = runtime.name
        self.stage = runtime.stage
        self.environment = runtime.environment