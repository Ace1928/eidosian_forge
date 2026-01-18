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
def WriteScenarioTest(command_tpl_name, collection_dict, test_output_dir):
    """Writes declarative YAML file for command.

  Args:
    command_tpl_name: name of command template file
    collection_dict: a mapping of collection info to feed template
    test_output_dir: path to directory in which to write YAML test file
  """
    test_tpl = _TemplateFileForCommandPath('scenario_unit_test_template.tpl', test=True)
    test_filename = command_tpl_name[:-len(TEMPLATE_SUFFIX)] + '.scenario.yaml'
    full_test_path = os.path.join(test_output_dir, test_filename)
    with files.FileWriter(full_test_path) as f:
        ctx = runtime.Context(f, **collection_dict)
        test_tpl.render_context(ctx)
    log.status.Print('New test written at ' + full_test_path)