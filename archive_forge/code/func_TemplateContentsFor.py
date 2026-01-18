from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.deployment_manager import importer
from googlecloudsdk.core import properties
def TemplateContentsFor(messages, template_path):
    """Build a TemplateContents message from a local template or url.

  Args:
    messages: The API message to use.
    template_path: Path to the config yaml file, with an optional list of
      imports.

  Returns:
    The TemplateContents message from the template at template_path.

  Raises:
    Error if the provided file is not a template.
  """
    config_obj = importer.BuildConfig(template=template_path)
    if not config_obj.IsTemplate():
        raise exceptions.Error('The provided file must be a template.')
    template_name = config_obj.GetBaseName()
    schema_name = template_name + '.schema'
    file_type = messages.TemplateContents.InterpreterValueValuesEnum.JINJA if template_name.endswith('.jinja') else messages.TemplateContents.InterpreterValueValuesEnum.PYTHON
    imports = importer.CreateImports(messages, config_obj)
    template = ''
    schema = ''
    for item in imports:
        if item.name == template_name:
            template = item.content
        elif item.name == schema_name:
            schema = item.content
    imports = [item for item in imports if item.name not in [template_name, schema_name]]
    return messages.TemplateContents(imports=imports, schema=schema, template=template, interpreter=file_type)