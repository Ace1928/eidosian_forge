from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
def _convert_from(self, proposed, errors='strict'):
    """Attempt to convert the markup to the proposed encoding.

        :param proposed: The name of a character encoding.
        """
    proposed = self.find_codec(proposed)
    if not proposed or (proposed, errors) in self.tried_encodings:
        return None
    self.tried_encodings.append((proposed, errors))
    markup = self.markup
    if self.smart_quotes_to is not None and proposed in self.ENCODINGS_WITH_SMART_QUOTES:
        smart_quotes_re = b'([\x80-\x9f])'
        smart_quotes_compiled = re.compile(smart_quotes_re)
        markup = smart_quotes_compiled.sub(self._sub_ms_char, markup)
    try:
        u = self._to_unicode(markup, proposed, errors)
        self.markup = u
        self.original_encoding = proposed
    except Exception as e:
        return None
    return self.markup