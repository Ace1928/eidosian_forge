from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.command_lib.help_search import lookup
from googlecloudsdk.core.document_renderers import render_document
import six
from six.moves import filter
def _AddFlagToSummary(self, location, terms):
    """Adds flag summary, given location such as ['flags']['--myflag']."""
    flags = self.command.get(location[0], {})
    line = ''
    assert len(location) > 2, self._IMPRECISE_LOCATION_MESSAGE.format(DOT.join(location))
    flag = flags.get(location[1])
    assert flag and (not flag[lookup.IS_HIDDEN]), self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
    if _FormatHeader(lookup.FLAGS) not in self._lines:
        self._lines.append(_FormatHeader(lookup.FLAGS))
    if _FormatItem(location[1]) not in self._lines:
        self._lines.append(_FormatItem(location[1]))
        desc_line = flag.get(lookup.DESCRIPTION, '')
        desc_line = _Snip(desc_line, self.length_per_snippet, terms)
        assert desc_line, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
        line = desc_line
    if location[2] == lookup.DEFAULT:
        default = flags.get(location[1]).get(lookup.DEFAULT)
        if default:
            if line not in self._lines:
                self._lines.append(line)
            if isinstance(default, dict):
                default = ', '.join([x for x in sorted(default.keys())])
            elif isinstance(default, list):
                default = ', '.join([x for x in default])
            line = 'Default: {}.'.format(default)
    else:
        valid_subattributes = [lookup.NAME, lookup.DESCRIPTION, lookup.CHOICES]
        assert location[2] in valid_subattributes, self._INVALID_LOCATION_MESSAGE.format(DOT.join(location))
    if line:
        self._lines.append(line)