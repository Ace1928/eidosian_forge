import re
from six.moves import html_entities as entities
import six
def _replace_entity(match):
    if match.group(1):
        ref = match.group(1)
        if ref.startswith('x'):
            ref = int(ref[1:], 16)
        else:
            ref = int(ref, 10)
        return six.unichr(ref)
    else:
        ref = match.group(2)
        if keepxmlentities and ref in ('amp', 'apos', 'gt', 'lt', 'quot'):
            return '&%s;' % ref
        try:
            return six.unichr(entities.name2codepoint[ref])
        except KeyError:
            if keepxmlentities:
                return '&amp;%s;' % ref
            else:
                return ref