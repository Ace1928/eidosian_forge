from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _GenerateDetailedHelpForCommand(resource, brief_doc_template, description_template, example_template):
    """Generates the detailed help doc for a command.

  Args:
    resource: The name of the resource. e.g "instance", "image" or "disk"
    brief_doc_template: The brief doc template to use.
    description_template: The command description template.
    example_template: The example template to use.
  Returns:
    The detailed help doc for a command. The returned value is a map with
    two attributes; 'brief' and 'description'.
  """
    product = _RESOURCE_NAME_TO_PRODUCT_NAME_MAP.get(resource, resource)
    product_plural = _PRODUCT_NAME_PLURAL_MAP.get(product, product + 's')
    sample = 'example-{0}'.format(resource)
    brief = brief_doc_template.format(product_plural)
    format_kwargs = {'product': product, 'sample': sample, 'resource': resource}
    description = description_template.format(**format_kwargs)
    examples = example_template.format(**format_kwargs) + _LIST_LABELS_DETAILED_HELP_TEMPLATE.format(**format_kwargs)
    return {'brief': brief, 'DESCRIPTION': description, 'EXAMPLES': examples}