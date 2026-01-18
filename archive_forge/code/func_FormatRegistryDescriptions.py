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
def FormatRegistryDescriptions():
    """Returns help markdown for all registered resource printer formats."""
    descriptions = ['The formats and format specific attributes are:\n']
    for name, printer in sorted(six.iteritems(resource_printer.GetFormatRegistry())):
        description, attributes, example = _ParseFormatDocString(printer)
        descriptions.append('\n*{name}*::\n{description}\n'.format(name=name, description=description))
        if attributes:
            _AppendParagraph(descriptions)
            descriptions.append('The format attributes are:\n\n')
            for attribute, description in attributes:
                descriptions.append('*{attribute}*:::\n{description}\n'.format(attribute=attribute, description=description))
            descriptions.append(':::\n')
        if example:
            _AppendParagraph(descriptions)
            descriptions.append('For example:\n+\n{example}\n'.format(example=''.join(example)))
    descriptions.append('::\n')
    description, attributes, example = _ParseFormatDocString(resource_printer.PrinterAttributes)
    if attributes:
        descriptions.append('\n{description}:\n+\n'.format(description=description[:-1]))
        for attribute, description in attributes:
            descriptions.append('*{attribute}*::\n{description}\n'.format(attribute=attribute, description=description))
    if example:
        _AppendParagraph(descriptions)
        descriptions.append('For example:\n+\n{example}\n'.format(example=''.join(example)))
    descriptions.append('\n')
    return ''.join(descriptions)