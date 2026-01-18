from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import progress_tracker
def ServiceDependencies():
    """Dependencies for the Service resource, for passing to ConditionPoller."""
    return {SERVICE_ROUTES_READY: {SERVICE_CONFIGURATIONS_READY}}