import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _format_help_text(description):
    """Format a comment for a policy based on the description provided.

    :param description: A string with helpful text.
    :returns: A line wrapped comment, or blank comment if description is None
    """
    if not description:
        return '#'
    formatted_lines = []
    paragraph = []

    def _wrap_paragraph(lines):
        return textwrap.wrap(' '.join(lines), 70, initial_indent='# ', subsequent_indent='# ')
    for line in description.strip().splitlines():
        if not line.strip():
            formatted_lines.extend(_wrap_paragraph(paragraph))
            formatted_lines.append('#')
            paragraph = []
        elif len(line) == len(line.lstrip()):
            paragraph.append(line.rstrip())
        else:
            if paragraph:
                warnings.warn('Invalid policy description: literal blocks must be preceded by a new line. This will raise an exception in a future version of oslo.policy:\n%s' % description, FutureWarning)
                formatted_lines.extend(_wrap_paragraph(paragraph))
                formatted_lines.append('#')
                paragraph = []
            formatted_lines.append('# %s' % line.rstrip())
    if paragraph:
        formatted_lines.extend(_wrap_paragraph(paragraph))
    return '\n'.join(formatted_lines)