from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidYamlError(exceptions.Error):
    """The Tekton Yaml user supplied is invalid."""

    def __init__(self, msg):
        msg = 'Invalid yaml: {msg}'.format(msg=msg)
        super(InvalidYamlError, self).__init__(msg)