from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
class ServiceConditionPoller(ConditionPoller):
    """A ConditionPoller for services."""

    def __init__(self, getter, tracker, dependencies=None, serv=None):
        super().__init__(getter, tracker, dependencies)
        self._resource_fail_type = serverless_exceptions.DeploymentFailedError