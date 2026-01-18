from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def ParseFile(self, input_file, file_format):
    if os.path.isdir(input_file):
        raise exceptions.BadFileException('[{0}] is a directory'.format(input_file))
    if not os.path.isfile(input_file):
        raise exceptions.BadFileException('No such file [{0}]'.format(input_file))
    try:
        with files.FileReader(input_file) as import_file:
            if file_format == 'json':
                return json.load(import_file)
            return yaml.load(import_file)
    except Exception as exp:
        msg = 'Unable to read route policy config from specified file [{0}] because {1}'.format(input_file, exp)
        raise exceptions.BadFileException(msg)