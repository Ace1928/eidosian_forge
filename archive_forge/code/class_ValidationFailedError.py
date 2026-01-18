from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.feature_flags import config
from googlecloudsdk.core.util import files
class ValidationFailedError(Exception):
    """Validation failed Error."""

    def __init__(self, config_file_path, config_file_errors, config_file_property_errors):
        msg_lines = []
        msg_lines.append('Invalid Feature Flag Config File\n[{}]\n'.format(config_file_path))
        for error in config_file_errors:
            msg_lines.append('{}: {}'.format(error.header, error.message))
        if config_file_property_errors:
            if config_file_errors:
                msg_lines.append('')
            msg_lines.append('PROPERTY ERRORS:')
        for section_property, errors in sorted(config_file_property_errors.items()):
            msg_lines.append('[{}]'.format(section_property))
            for error in errors:
                msg_lines.append('\t{}: {}'.format(error.header, error.message))
        super(ValidationFailedError, self).__init__('\n'.join(msg_lines))