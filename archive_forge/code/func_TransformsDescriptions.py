from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def TransformsDescriptions(transforms):
    """Generates resource transform help text markdown for transforms.

  Args:
    transforms: The transform name=>method symbol table.

  Returns:
    The resource transform help text markdown for transforms.
  """
    descriptions = []
    for name, transform in sorted(six.iteritems(transforms)):
        description, prototype, args, example = _ParseTransformDocString(transform)
        if not description:
            continue
        descriptions.append('\n\n*{name}*{prototype}::\n{description}'.format(name=name, prototype=prototype, description=description))
        if args:
            _AppendParagraph(descriptions)
            descriptions.append('The arguments are:\n+\n')
            for arg, description in args:
                descriptions.append('*```{arg}```*:::\n{description}\n'.format(arg=arg, description=description))
                descriptions.append(':::\n')
        if example:
            _AppendParagraph(descriptions)
            descriptions.append('For example:\n+\n{example}\n'.format(example=''.join(example)))
        descriptions.append('::\n')
    return ''.join(descriptions)