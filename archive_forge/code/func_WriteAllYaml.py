from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def WriteAllYaml(collection_name, output_dir):
    """Writes declarative YAML file for all supported command types.

  Args:
    collection_name: name of collection to generate commands for.
    output_dir: path to the directory where generated YAML files will be
      written.
  """
    collection_dict = _MakeCollectionDict(collection_name)
    api_message_module = apis.GetMessagesModule(collection_dict['api_name'], collection_dict['api_version'])
    api_dict = _MakeApiDict(api_message_module, collection_dict)
    collection_dict.update(api_dict)
    for command_template in os.listdir(os.path.join(os.path.dirname(__file__), 'command_templates')):
        if command_template.split('/')[-1] not in CRUD_TEMPLATES:
            continue
        should_write_test = WriteYaml(command_template, collection_dict, output_dir, api_message_module)
        if should_write_test:
            WriteScenarioTest(command_template, collection_dict, output_dir)