from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
class SecurityCenterSettingsException(core_exceptions.Error):
    """Exception raised from SCC settings backend api."""
    pass