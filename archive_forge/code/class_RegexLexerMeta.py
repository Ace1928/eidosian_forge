from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
class RegexLexerMeta(LexerMeta):
    """
    Metaclass for RegexLexer, creates the self._tokens attribute from
    self.tokens on the first instantiation.
    """

    def _process_regex(cls, regex, rflags, state):
        """Preprocess the regular expression component of a token definition."""
        if isinstance(regex, Future):
            regex = regex.get()
        return re.compile(regex, rflags).match

    def _process_token(cls, token):
        """Preprocess the token component of a token definition."""
        assert type(token) is _TokenType or callable(token), 'token type must be simple type or callable, not %r' % (token,)
        return token

    def _process_new_state(cls, new_state, unprocessed, processed):
        """Preprocess the state transition action of a token definition."""
        if isinstance(new_state, str):
            if new_state == '#pop':
                return -1
            elif new_state in unprocessed:
                return (new_state,)
            elif new_state == '#push':
                return new_state
            elif new_state[:5] == '#pop:':
                return -int(new_state[5:])
            else:
                assert False, 'unknown new state %r' % new_state
        elif isinstance(new_state, combined):
            tmp_state = '_tmp_%d' % cls._tmpname
            cls._tmpname += 1
            itokens = []
            for istate in new_state:
                assert istate != new_state, 'circular state ref %r' % istate
                itokens.extend(cls._process_state(unprocessed, processed, istate))
            processed[tmp_state] = itokens
            return (tmp_state,)
        elif isinstance(new_state, tuple):
            for istate in new_state:
                assert istate in unprocessed or istate in ('#pop', '#push'), 'unknown new state ' + istate
            return new_state
        else:
            assert False, 'unknown new state def %r' % new_state

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

    def process_tokendef(cls, name, tokendefs=None):
        """Preprocess a dictionary of token definitions."""
        processed = cls._all_tokens[name] = {}
        tokendefs = tokendefs or cls.tokens[name]
        for state in list(tokendefs):
            cls._process_state(tokendefs, processed, state)
        return processed

    def get_tokendefs(cls):
        """
        Merge tokens from superclasses in MRO order, returning a single tokendef
        dictionary.

        Any state that is not defined by a subclass will be inherited
        automatically.  States that *are* defined by subclasses will, by
        default, override that state in the superclass.  If a subclass wishes to
        inherit definitions from a superclass, it can use the special value
        "inherit", which will cause the superclass' state definition to be
        included at that point in the state.
        """
        tokens = {}
        inheritable = {}
        for c in cls.__mro__:
            toks = c.__dict__.get('tokens', {})
            for state, items in iteritems(toks):
                curitems = tokens.get(state)
                if curitems is None:
                    tokens[state] = items
                    try:
                        inherit_ndx = items.index(inherit)
                    except ValueError:
                        continue
                    inheritable[state] = inherit_ndx
                    continue
                inherit_ndx = inheritable.pop(state, None)
                if inherit_ndx is None:
                    continue
                curitems[inherit_ndx:inherit_ndx + 1] = items
                try:
                    new_inh_ndx = items.index(inherit)
                except ValueError:
                    pass
                else:
                    inheritable[state] = inherit_ndx + new_inh_ndx
        return tokens

    def __call__(cls, *args, **kwds):
        """Instantiate cls after preprocessing its token definitions."""
        if '_tokens' not in cls.__dict__:
            cls._all_tokens = {}
            cls._tmpname = 0
            if hasattr(cls, 'token_variants') and cls.token_variants:
                pass
            else:
                cls._tokens = cls.process_tokendef('', cls.get_tokendefs())
        return type.__call__(cls, *args, **kwds)