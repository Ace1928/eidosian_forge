from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
import jsonschema
def ValidateWithDetailedError(self, parsed_yaml):
    """Validates parsed_yaml against JSON schema.

    Provides details of validation failure in the returned error message.
    Args:
      parsed_yaml: YAML to validate

    Raises:
      ValidationError: if the template doesn't obey the schema.
    """
    try:
        self._validate(parsed_yaml)
    except jsonschema.RefResolutionError as e:
        raise RefError(e)
    except jsonschema.exceptions.ValidationError as ve:
        msg = io.StringIO()
        msg.write('ERROR: Schema validation failed: {}\n\n'.format(ve))
        if ve.cause:
            additional_exception = 'Root Exception: {}'.format(ve.cause)
        else:
            additional_exception = ''
        root_error = ve.context[-1] if ve.context else None
        if root_error:
            error_path = ''.join(('[{}]'.format(elem) for elem in root_error.absolute_path))
        else:
            error_path = ''
        msg.write('Additional Details:\nError Message: {msg}\n\nFailing Validation Schema: {schema}\n\nFailing Element: {instance}\n\nFailing Element Path: {path}\n\n{additional_cause}\n'.format(msg=root_error.message if root_error else None, instance=root_error.instance if root_error else None, schema=root_error.schema if root_error else None, path=error_path, additional_cause=additional_exception))
        ve.message = msg.getvalue()
        raise ValidationError(ve)