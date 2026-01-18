from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ConflictPolicyException(core_exceptions.Error):
    """For conflict policies from inputs."""

    def __init__(self, parameter_names):
        super(ConflictPolicyException, self).__init__('Invalid value for [{0}]: Please make sure {0} resources are all from the same policy.'.format(', '.join(['{0}'.format(p) for p in parameter_names])))