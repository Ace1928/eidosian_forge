from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
def _ExtractHelpStrings(self, docstring):
    """Extracts short help, long help and man page index from a docstring.

    Sets self.short_help, self.long_help and self.index_help and adds release
    track tags if needed.

    Args:
      docstring: The docstring from which short and long help are to be taken
    """
    self.short_help, self.long_help = usage_text.ExtractHelpStrings(docstring)
    if 'brief' in self.detailed_help:
        self.short_help = re.sub('\\s', ' ', self.detailed_help['brief']).strip()
    if self.short_help and (not self.short_help.endswith('.')):
        self.short_help += '.'
    if self.Notices():
        all_notices = '\n\n' + '\n\n'.join(sorted(self.Notices().values())) + '\n\n'
        description = self.detailed_help.get('DESCRIPTION')
        if description:
            self.detailed_help = dict(self.detailed_help)
            self.detailed_help['DESCRIPTION'] = all_notices + textwrap.dedent(description)
        if self.short_help == self.long_help:
            self.long_help += all_notices
        else:
            self.long_help = self.short_help + all_notices + self.long_help
    self.index_help = self.short_help
    if len(self.index_help) > 1:
        if self.index_help[0].isupper() and (not self.index_help[1].isupper()):
            self.index_help = self.index_help[0].lower() + self.index_help[1:]
        if self.index_help[-1] == '.':
            self.index_help = self.index_help[:-1]
    tags = []
    tag = self.ReleaseTrack().help_tag
    if tag:
        tags.append(tag)
    if self.Notices():
        tags.extend(sorted(self.Notices().keys()))
    if tags:
        tag = ' '.join(tags) + ' '

        def _InsertTag(txt):
            return re.sub('^(\\s*)', '\\1' + tag, txt)
        self.short_help = _InsertTag(self.short_help)
        if not self.long_help.startswith('#'):
            self.long_help = _InsertTag(self.long_help)
        description = self.detailed_help.get('DESCRIPTION')
        if description and (not re.match('^[ \\n]*\\{(description|index)\\}', description)):
            self.detailed_help = dict(self.detailed_help)
            self.detailed_help['DESCRIPTION'] = _InsertTag(textwrap.dedent(description))