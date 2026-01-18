from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class AutomationNameFormatError(exceptions.Error):
    """Error when the name of the automation in the config file is not formatted correctly."""

    def __init__(self, automation_name):
        super(AutomationNameFormatError, self).__init__('Automation name {} in the configuration should be in the formatof pipeline_id/automation_id.'.format(automation_name))