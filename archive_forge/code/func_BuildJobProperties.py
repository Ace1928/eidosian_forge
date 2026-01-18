from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def BuildJobProperties(arg_properties, properties_file):
    """Build job properties.

  Merges properties from the arg_properties and properties_file. If a property
  is set in both, the value in arg_properties is used.

  Args:
    arg_properties: A dictionary of property=value pairs.
    properties_file: Path or URI to a text file with property=value lines
    and/or comments. File can be a local file or a gs:// file.

  Returns:
    A dictionary merged properties

  Example:
    BuildJobProperties({'foo':'bar'}, 'gs://test-bucket/job_properties.conf')
  """
    job_properties = {}
    if properties_file:
        try:
            if properties_file.startswith('gs://'):
                data = storage_helpers.ReadObject(properties_file)
            else:
                data = console_io.ReadFromFileOrStdin(properties_file, binary=False)
        except Exception as e:
            raise exceptions.Error('Cannot read properties-file: {0}'.format(e))
        try:
            yaml.allow_duplicate_keys = True
            key_values = yaml.load(data.strip().replace('=', ': '), round_trip=True)
            if key_values:
                for key, value in key_values.items():
                    job_properties[key] = value
        except Exception:
            raise exceptions.ParseError('Cannot parse properties-file: {0}, '.format(properties_file) + 'make sure file format is a text file with list of key=value')
    if arg_properties:
        job_properties.update(arg_properties)
    return job_properties