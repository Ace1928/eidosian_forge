from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def ParseTemplate(template_file, params=None, params_from_file=None):
    """Parse and apply params into a template file.

  Args:
    template_file: The path to the file to open and parse.
    params: a dict of param-name -> param-value
    params_from_file: a dict of param-name -> param-file

  Returns:
    The parsed template dict

  Raises:
    yaml.Error: When the template file cannot be read or parsed.
    ArgumentError: If any params are not provided.
    ValidationError: if the YAML file is invalid.
  """
    params = params or {}
    params_from_file = params_from_file or {}
    joined_params = dict(params)
    for key, file_path in six.iteritems(params_from_file):
        if key in joined_params:
            raise exceptions.DuplicateError('Duplicate param key: ' + key)
        try:
            joined_params[key] = files.ReadFileContents(file_path)
        except files.Error as e:
            raise exceptions.ArgumentError('Could not load param key "{0}" from file "{1}": {2}'.format(key, file_path, e.strerror))
    template = yaml.load_path(template_file)
    if not isinstance(template, dict) or 'template' not in template:
        raise exceptions.ValidationError('Invalid template format.  Root must be a mapping with single "template" value')
    template, missing_params, used_params = ReplaceTemplateParams(template, joined_params)
    if missing_params:
        raise exceptions.ArgumentError('Some parameters were present in the template but not provided on the command line: ' + ', '.join(sorted(missing_params)))
    unused_params = set(joined_params.keys()) - used_params
    if unused_params:
        raise exceptions.ArgumentError('Some parameters were specified on the command line but not referenced in the template: ' + ', '.join(sorted(unused_params)))
    return template