import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
class FeatStructReader:

    def __init__(self, features=(SLASH, TYPE), fdict_class=FeatStruct, flist_class=FeatList, logic_parser=None):
        self._features = {f.name: f for f in features}
        self._fdict_class = fdict_class
        self._flist_class = flist_class
        self._prefix_feature = None
        self._slash_feature = None
        for feature in features:
            if feature.display == 'slash':
                if self._slash_feature:
                    raise ValueError('Multiple features w/ display=slash')
                self._slash_feature = feature
            if feature.display == 'prefix':
                if self._prefix_feature:
                    raise ValueError('Multiple features w/ display=prefix')
                self._prefix_feature = feature
        self._features_with_defaults = [feature for feature in features if feature.default is not None]
        if logic_parser is None:
            logic_parser = LogicParser()
        self._logic_parser = logic_parser

    def fromstring(self, s, fstruct=None):
        """
        Convert a string representation of a feature structure (as
        displayed by repr) into a ``FeatStruct``.  This process
        imposes the following restrictions on the string
        representation:

        - Feature names cannot contain any of the following:
          whitespace, parentheses, quote marks, equals signs,
          dashes, commas, and square brackets.  Feature names may
          not begin with plus signs or minus signs.
        - Only the following basic feature value are supported:
          strings, integers, variables, None, and unquoted
          alphanumeric strings.
        - For reentrant values, the first mention must specify
          a reentrance identifier and a value; and any subsequent
          mentions must use arrows (``'->'``) to reference the
          reentrance identifier.
        """
        s = s.strip()
        value, position = self.read_partial(s, 0, {}, fstruct)
        if position != len(s):
            self._error(s, 'end of string', position)
        return value
    _START_FSTRUCT_RE = re.compile('\\s*(?:\\((\\d+)\\)\\s*)?(\\??[\\w-]+)?(\\[)')
    _END_FSTRUCT_RE = re.compile('\\s*]\\s*')
    _SLASH_RE = re.compile('/')
    _FEATURE_NAME_RE = re.compile('\\s*([+-]?)([^\\s\\(\\)<>"\\\'\\-=\\[\\],]+)\\s*')
    _REENTRANCE_RE = re.compile('\\s*->\\s*')
    _TARGET_RE = re.compile('\\s*\\((\\d+)\\)\\s*')
    _ASSIGN_RE = re.compile('\\s*=\\s*')
    _COMMA_RE = re.compile('\\s*,\\s*')
    _BARE_PREFIX_RE = re.compile('\\s*(?:\\((\\d+)\\)\\s*)?(\\??[\\w-]+\\s*)()')
    _START_FDICT_RE = re.compile('(%s)|(%s\\s*(%s\\s*(=|->)|[+-]%s|\\]))' % (_BARE_PREFIX_RE.pattern, _START_FSTRUCT_RE.pattern, _FEATURE_NAME_RE.pattern, _FEATURE_NAME_RE.pattern))

    def read_partial(self, s, position=0, reentrances=None, fstruct=None):
        """
        Helper function that reads in a feature structure.

        :param s: The string to read.
        :param position: The position in the string to start parsing.
        :param reentrances: A dictionary from reentrance ids to values.
            Defaults to an empty dictionary.
        :return: A tuple (val, pos) of the feature structure created by
            parsing and the position where the parsed feature structure ends.
        :rtype: bool
        """
        if reentrances is None:
            reentrances = {}
        try:
            return self._read_partial(s, position, reentrances, fstruct)
        except ValueError as e:
            if len(e.args) != 2:
                raise
            self._error(s, *e.args)

    def _read_partial(self, s, position, reentrances, fstruct=None):
        if fstruct is None:
            if self._START_FDICT_RE.match(s, position):
                fstruct = self._fdict_class()
            else:
                fstruct = self._flist_class()
        match = self._START_FSTRUCT_RE.match(s, position)
        if not match:
            match = self._BARE_PREFIX_RE.match(s, position)
            if not match:
                raise ValueError('open bracket or identifier', position)
        position = match.end()
        if match.group(1):
            identifier = match.group(1)
            if identifier in reentrances:
                raise ValueError('new identifier', match.start(1))
            reentrances[identifier] = fstruct
        if isinstance(fstruct, FeatDict):
            fstruct.clear()
            return self._read_partial_featdict(s, position, match, reentrances, fstruct)
        else:
            del fstruct[:]
            return self._read_partial_featlist(s, position, match, reentrances, fstruct)

    def _read_partial_featlist(self, s, position, match, reentrances, fstruct):
        if match.group(2):
            raise ValueError('open bracket')
        if not match.group(3):
            raise ValueError('open bracket')
        while position < len(s):
            match = self._END_FSTRUCT_RE.match(s, position)
            if match is not None:
                return (fstruct, match.end())
            match = self._REENTRANCE_RE.match(s, position)
            if match:
                position = match.end()
                match = self._TARGET_RE.match(s, position)
                if not match:
                    raise ValueError('identifier', position)
                target = match.group(1)
                if target not in reentrances:
                    raise ValueError('bound identifier', position)
                position = match.end()
                fstruct.append(reentrances[target])
            else:
                value, position = self._read_value(0, s, position, reentrances)
                fstruct.append(value)
            if self._END_FSTRUCT_RE.match(s, position):
                continue
            match = self._COMMA_RE.match(s, position)
            if match is None:
                raise ValueError('comma', position)
            position = match.end()
        raise ValueError('close bracket', position)

    def _read_partial_featdict(self, s, position, match, reentrances, fstruct):
        if match.group(2):
            if self._prefix_feature is None:
                raise ValueError('open bracket or identifier', match.start(2))
            prefixval = match.group(2).strip()
            if prefixval.startswith('?'):
                prefixval = Variable(prefixval)
            fstruct[self._prefix_feature] = prefixval
        if not match.group(3):
            return self._finalize(s, match.end(), reentrances, fstruct)
        while position < len(s):
            name = value = None
            match = self._END_FSTRUCT_RE.match(s, position)
            if match is not None:
                return self._finalize(s, match.end(), reentrances, fstruct)
            match = self._FEATURE_NAME_RE.match(s, position)
            if match is None:
                raise ValueError('feature name', position)
            name = match.group(2)
            position = match.end()
            if name[0] == '*' and name[-1] == '*':
                name = self._features.get(name[1:-1])
                if name is None:
                    raise ValueError('known special feature', match.start(2))
            if name in fstruct:
                raise ValueError('new name', match.start(2))
            if match.group(1) == '+':
                value = True
            if match.group(1) == '-':
                value = False
            if value is None:
                match = self._REENTRANCE_RE.match(s, position)
                if match is not None:
                    position = match.end()
                    match = self._TARGET_RE.match(s, position)
                    if not match:
                        raise ValueError('identifier', position)
                    target = match.group(1)
                    if target not in reentrances:
                        raise ValueError('bound identifier', position)
                    position = match.end()
                    value = reentrances[target]
            if value is None:
                match = self._ASSIGN_RE.match(s, position)
                if match:
                    position = match.end()
                    value, position = self._read_value(name, s, position, reentrances)
                else:
                    raise ValueError('equals sign', position)
            fstruct[name] = value
            if self._END_FSTRUCT_RE.match(s, position):
                continue
            match = self._COMMA_RE.match(s, position)
            if match is None:
                raise ValueError('comma', position)
            position = match.end()
        raise ValueError('close bracket', position)

    def _finalize(self, s, pos, reentrances, fstruct):
        """
        Called when we see the close brace -- checks for a slash feature,
        and adds in default values.
        """
        match = self._SLASH_RE.match(s, pos)
        if match:
            name = self._slash_feature
            v, pos = self._read_value(name, s, match.end(), reentrances)
            fstruct[name] = v
        return (fstruct, pos)

    def _read_value(self, name, s, position, reentrances):
        if isinstance(name, Feature):
            return name.read_value(s, position, reentrances, self)
        else:
            return self.read_value(s, position, reentrances)

    def read_value(self, s, position, reentrances):
        for handler, regexp in self.VALUE_HANDLERS:
            match = regexp.match(s, position)
            if match:
                handler_func = getattr(self, handler)
                return handler_func(s, position, reentrances, match)
        raise ValueError('value', position)

    def _error(self, s, expected, position):
        lines = s.split('\n')
        while position > len(lines[0]):
            position -= len(lines.pop(0)) + 1
        estr = 'Error parsing feature structure\n    ' + lines[0] + '\n    ' + ' ' * position + '^ ' + 'Expected %s' % expected
        raise ValueError(estr)
    VALUE_HANDLERS = [('read_fstruct_value', _START_FSTRUCT_RE), ('read_var_value', re.compile('\\?[a-zA-Z_][a-zA-Z0-9_]*')), ('read_str_value', re.compile('[uU]?[rR]?([\'"])')), ('read_int_value', re.compile('-?\\d+')), ('read_sym_value', re.compile('[a-zA-Z_][a-zA-Z0-9_]*')), ('read_app_value', re.compile('<(app)\\((\\?[a-z][a-z]*)\\s*,\\s*(\\?[a-z][a-z]*)\\)>')), ('read_logic_value', re.compile('<(.*?)(?<!-)>')), ('read_set_value', re.compile('{')), ('read_tuple_value', re.compile('\\('))]

    def read_fstruct_value(self, s, position, reentrances, match):
        return self.read_partial(s, position, reentrances)

    def read_str_value(self, s, position, reentrances, match):
        return read_str(s, position)

    def read_int_value(self, s, position, reentrances, match):
        return (int(match.group()), match.end())

    def read_var_value(self, s, position, reentrances, match):
        return (Variable(match.group()), match.end())
    _SYM_CONSTS = {'None': None, 'True': True, 'False': False}

    def read_sym_value(self, s, position, reentrances, match):
        val, end = (match.group(), match.end())
        return (self._SYM_CONSTS.get(val, val), end)

    def read_app_value(self, s, position, reentrances, match):
        """Mainly included for backwards compat."""
        return (self._logic_parser.parse('%s(%s)' % match.group(2, 3)), match.end())

    def read_logic_value(self, s, position, reentrances, match):
        try:
            try:
                expr = self._logic_parser.parse(match.group(1))
            except LogicalExpressionException as e:
                raise ValueError from e
            return (expr, match.end())
        except ValueError as e:
            raise ValueError('logic expression', match.start(1)) from e

    def read_tuple_value(self, s, position, reentrances, match):
        return self._read_seq_value(s, position, reentrances, match, ')', FeatureValueTuple, FeatureValueConcat)

    def read_set_value(self, s, position, reentrances, match):
        return self._read_seq_value(s, position, reentrances, match, '}', FeatureValueSet, FeatureValueUnion)

    def _read_seq_value(self, s, position, reentrances, match, close_paren, seq_class, plus_class):
        """
        Helper function used by read_tuple_value and read_set_value.
        """
        cp = re.escape(close_paren)
        position = match.end()
        m = re.compile('\\s*/?\\s*%s' % cp).match(s, position)
        if m:
            return (seq_class(), m.end())
        values = []
        seen_plus = False
        while True:
            m = re.compile('\\s*%s' % cp).match(s, position)
            if m:
                if seen_plus:
                    return (plus_class(values), m.end())
                else:
                    return (seq_class(values), m.end())
            val, position = self.read_value(s, position, reentrances)
            values.append(val)
            m = re.compile('\\s*(,|\\+|(?=%s))\\s*' % cp).match(s, position)
            if not m:
                raise ValueError("',' or '+' or '%s'" % cp, position)
            if m.group(1) == '+':
                seen_plus = True
            position = m.end()