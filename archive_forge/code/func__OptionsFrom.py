from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
def _OptionsFrom(messages, options_data):
    """Translate a dict of options data into a message object.

  Args:
    messages: The API message to use.
    options_data: A dict containing options data.

  Returns:
    An Options message object derived from options_data.
  """
    options = messages.Options()
    if 'virtualProperties' in options_data:
        options.virtualProperties = options_data['virtualProperties']
    if 'inputMappings' in options_data:
        options.inputMappings = [_InputMappingFrom(messages, im_data) for im_data in options_data['inputMappings']]
    if 'validationOptions' in options_data:
        validation_options_data = options_data['validationOptions']
        validation_options = messages.ValidationOptions()
        if 'schemaValidation' in validation_options_data:
            validation_options.schemaValidation = messages.ValidationOptions.SchemaValidationValueValuesEnum(validation_options_data['schemaValidation'])
        if 'undeclaredProperties' in validation_options_data:
            validation_options.undeclaredProperties = messages.ValidationOptions.UndeclaredPropertiesValueValuesEnum(validation_options_data['undeclaredProperties'])
        options.validationOptions = validation_options
    return options