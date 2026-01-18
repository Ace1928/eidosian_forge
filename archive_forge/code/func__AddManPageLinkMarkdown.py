from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _AddManPageLinkMarkdown(self, doc):
    """Add <top> ...(1) man page link markdown to doc."""
    if not self._command_path:
        return doc
    pat = re.compile('(\\*?(' + self._top + '(?:[-_ a-z])*)\\*?)\\(1\\)')
    pos = 0
    rep = ''
    while True:
        match = pat.search(doc, pos)
        if not match:
            break
        cmd = match.group(2).replace('_', ' ')
        ref = cmd.replace(' ', '/')
        lnk = '*link:' + ref + '[' + cmd + ']*'
        rep += doc[pos:match.start(2)] + lnk
        pos = match.end(1)
    if rep:
        doc = rep + doc[pos:]
    return doc