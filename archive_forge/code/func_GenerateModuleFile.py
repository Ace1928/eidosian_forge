from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.calliope.exceptions import core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
from mako import runtime
from mako import template
def GenerateModuleFile(import_data, project, dest_file=None, dest_dir=None):
    """Generate terraform modules file from template."""
    output_file_name = os.path.join(dest_dir, dest_file)
    output_template = _BuildTemplate('terraform_module_template.tpl')
    module_contents = set()
    for import_path, _ in import_data:
        module_source, module_name = ConstructModuleParameters(import_path, dest_dir)
        module_contents.add((module_name, module_source))
    module_declarations = []
    for module in module_contents:
        module_declarations.append(MODULE_TEMPLATE.format(module_name=module[0], module_source=module[1]))
    context = {'project': project}
    context['modules'] = os.linesep.join(module_declarations)
    try:
        with files.FileWriter(output_file_name, create_path=True) as f:
            ctx = runtime.Context(f, **context)
            output_template.render_context(ctx)
        os.chmod(output_file_name, 493)
    except files.Error as e:
        raise TerraformGenerationError('Error writing import script::{}'.format(e))
    return (output_file_name, len(module_contents))