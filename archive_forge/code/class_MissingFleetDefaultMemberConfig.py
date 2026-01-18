from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingFleetDefaultMemberConfig(exceptions.Error):
    """For when the fleet default member config is required but missing."""