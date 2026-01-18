from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def _process_state(cls, unprocessed, processed, state):
    """Preprocess a single state definition."""
    assert type(state) is str, 'wrong state name %r' % state
    assert state[0] != '#', 'invalid state name %r' % state
    if state in processed:
        return processed[state]
    tokens = processed[state] = []
    rflags = cls.flags
    for tdef in unprocessed[state]:
        if isinstance(tdef, include):
            assert tdef != state, 'circular state reference %r' % state
            tokens.extend(cls._process_state(unprocessed, processed, str(tdef)))
            continue
        if isinstance(tdef, _inherit):
            continue
        if isinstance(tdef, default):
            new_state = cls._process_new_state(tdef.state, unprocessed, processed)
            tokens.append((re.compile('').match, None, new_state))
            continue
        assert type(tdef) is tuple, 'wrong rule def %r' % tdef
        try:
            rex = cls._process_regex(tdef[0], rflags, state)
        except Exception as err:
            raise ValueError('uncompilable regex %r in state %r of %r: %s' % (tdef[0], state, cls, err))
        token = cls._process_token(tdef[1])
        if len(tdef) == 2:
            new_state = None
        else:
            new_state = cls._process_new_state(tdef[2], unprocessed, processed)
        tokens.append((rex, token, new_state))
    return tokens