from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _RenderTestFiles(output_root, resource_data, collection_info, enable_overwrites):
    """Render python test file using template and context."""
    context = _BuildTestContext(collection_info, resource_data)
    init_path = _BuildFilePath(output_root, _TEST_PATH_COMPONENTS, resource_data.home_directory, '__init__.py')
    init_template = _BuildTemplate('python_blank_init_template.tpl')
    _RenderFile(init_path, init_template, context, enable_overwrites)
    test_path = _BuildFilePath(output_root, _TEST_PATH_COMPONENTS, resource_data.home_directory, 'config_export_test.py')
    test_template = _BuildTemplate('unit_test_template.tpl')
    _RenderFile(test_path, test_template, context, enable_overwrites)