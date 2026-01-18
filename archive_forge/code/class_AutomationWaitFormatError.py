from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class AutomationWaitFormatError(exceptions.Error):
    """Error when the name of the automation in the config file is not formatted correctly."""

    def __init__(self):
        super(AutomationWaitFormatError, self).__init__('Wait must be numbers with the last character m, e.g. 5m.')