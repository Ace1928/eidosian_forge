import logging
import re
from oslo_policy import _checks
def _parse_tokenize(rule):
    """Tokenizer for the policy language.

    Most of the single-character tokens are specified in the
    _tokenize_re; however, parentheses need to be handled specially,
    because they can appear inside a check string.  Thankfully, those
    parentheses that appear inside a check string can never occur at
    the very beginning or end ("%(variable)s" is the correct syntax).
    """
    for tok in _tokenize_re.split(rule):
        if not tok or tok.isspace():
            continue
        clean = tok.lstrip('(')
        for i in range(len(tok) - len(clean)):
            yield ('(', '(')
        if not clean:
            continue
        else:
            tok = clean
        clean = tok.rstrip(')')
        trail = len(tok) - len(clean)
        lowered = clean.lower()
        if lowered in ('and', 'or', 'not'):
            yield (lowered, clean)
        elif clean:
            if len(tok) >= 2 and (tok[0], tok[-1]) in [('"', '"'), ("'", "'")]:
                yield ('string', tok[1:-1])
            else:
                yield ('check', _parse_check(clean))
        for i in range(trail):
            yield (')', ')')