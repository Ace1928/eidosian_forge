from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def ValidateFieldConfig(unused_ref, args, request):
    """Python hook to validate the field configuration of the given request.

  Note that this hook is only called after the request has been formed based on
  the spec. Thus, the validation of the user's choices for order and
  array-config, as well as the check for the required field-path attribute, have
  already been performed. As such the only remaining things to verify are that
  the user has specified at least 2 fields, and that exactly one of order or
  array-config was specified for each field.

  Args:
    unused_ref: The resource ref (unused).
    args: The parsed arg namespace.
    request: The request formed based on the spec.
  Returns:
    The original request assuming the field configuration is valid.
  Raises:
    InvalidArgumentException: If the field configuration is invalid.
  """
    if len(args.field_config) == 1 and args.field_config[0].vectorConfig:
        pass
    elif len(args.field_config) < 2:
        raise exceptions.InvalidArgumentException('--field-config', 'Composite indexes must be configured with at least 2 fields. For single-field index management, use the commands under `gcloud firestore indexes fields`.')
    invalid_field_configs = []
    for field_config in args.field_config:
        order = field_config.order
        array_config = field_config.arrayConfig
        if field_config.vectorConfig:
            continue
        if order and array_config or (not order and (not array_config)):
            invalid_field_configs.append(field_config)
    if invalid_field_configs:
        raise exceptions.InvalidArgumentException('--field-config', "Exactly one of 'order' or 'array-config' must be specified for the {field_word} with the following {path_word}: [{paths}].".format(field_word=text.Pluralize(len(invalid_field_configs), 'field'), path_word=text.Pluralize(len(invalid_field_configs), 'path'), paths=', '.join((field_config.fieldPath for field_config in invalid_field_configs))))
    return request