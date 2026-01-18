from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingAttachedInstallAgent(exceptions.Error):
    """Class for errors by missing attached cluster install agent."""

    def __init__(self, extra_message=None):
        message = 'Missing attached cluster install agent.'
        if extra_message:
            message += ' ' + extra_message
        super(MissingAttachedInstallAgent, self).__init__(message)