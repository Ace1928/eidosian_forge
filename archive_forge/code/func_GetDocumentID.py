from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.core.document_renderers import renderer
def GetDocumentID(self, name):
    """Returns a unique document id for name."""

    def SingularizeCommandAndGroupDocumentID(name):
        """Returns singlularized name if name is 'COMMANDS' or 'GROUPS'."""
        return re.sub('(COMMAND|GROUP)S$', '\\1', name)
    m = re.match('(-- |\\[)*(<[^>]*>)*(?P<anchor>-[-_a-z0-9\\[\\]]+|[_A-Za-z.0-9 ][-_A-Za-z.0-9 ]*|[-.0-9]+).*', name)
    if m:
        name = m.group('anchor')
    name = name.strip(' ').replace(' ', '-')
    name = SingularizeCommandAndGroupDocumentID(name)
    attempt = name
    number = 0
    while True:
        if attempt not in self._document_ids:
            self._document_ids.add(attempt)
            return attempt
        number += 1
        attempt = '{name}-{number}'.format(name=name, number=number)