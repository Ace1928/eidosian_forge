from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class ExecutableCredConfigGenerator(CredConfigGenerator):
    """The generator for executable-command-based credentials configs."""

    def __init__(self, config_type, command, timeout_millis, output_file):
        if timeout_millis:
            timeout_millis = int(timeout_millis)
        super(ExecutableCredConfigGenerator, self).__init__(config_type)
        self.command = command
        self.timeout_millis = timeout_millis or 30000
        self.output_file = output_file

    def get_source(self, args):
        executable_config = {'command': self.command, 'timeout_millis': self.timeout_millis}
        if self.output_file:
            executable_config['output_file'] = self.output_file
        return {'executable': executable_config}