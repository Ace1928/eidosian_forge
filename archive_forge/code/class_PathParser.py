from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
class PathParser(object):
    """Tokenizes and parses an XPath expression."""
    _QUOTES = (("'", "'"), ('"', '"'))
    _TOKENS = ('::', ':', '..', '.', '//', '/', '[', ']', '()', '(', ')', '@', '=', '!=', '!', '|', ',', '>=', '>', '<=', '<', '$')
    _tokenize = re.compile('("[^"]*")|(\\\'[^\\\']*\\\')|((?:\\d+)?\\.\\d+)|(%s)|([^%s\\s]+)|\\s+' % ('|'.join([re.escape(t) for t in _TOKENS]), ''.join([re.escape(t[0]) for t in _TOKENS]))).findall

    def __init__(self, text, filename=None, lineno=-1):
        self.filename = filename
        self.lineno = lineno
        self.tokens = [t for t in [dqstr or sqstr or number or token or name for dqstr, sqstr, number, token, name in self._tokenize(text)] if t]
        self.pos = 0

    @property
    def at_end(self):
        return self.pos == len(self.tokens) - 1

    @property
    def cur_token(self):
        return self.tokens[self.pos]

    def next_token(self):
        self.pos += 1
        return self.tokens[self.pos]

    def peek_token(self):
        if not self.at_end:
            return self.tokens[self.pos + 1]
        return None

    def parse(self):
        """Parses the XPath expression and returns a list of location path
        tests.
        
        For union expressions (such as `*|text()`), this function returns one
        test for each operand in the union. For patch expressions that don't
        use the union operator, the function always returns a list of size 1.
        
        Each path test in turn is a sequence of tests that correspond to the
        location steps, each tuples of the form `(axis, testfunc, predicates)`
        """
        paths = [self._location_path()]
        while self.cur_token == '|':
            self.next_token()
            paths.append(self._location_path())
        if not self.at_end:
            raise PathSyntaxError('Unexpected token %r after end of expression' % self.cur_token, self.filename, self.lineno)
        return paths

    def _location_path(self):
        steps = []
        while True:
            if self.cur_token.startswith('/'):
                if not steps:
                    if self.cur_token == '//':
                        self.next_token()
                        axis, nodetest, predicates = self._location_step()
                        steps.append((DESCENDANT_OR_SELF, nodetest, predicates))
                        if self.at_end or not self.cur_token.startswith('/'):
                            break
                        continue
                    else:
                        raise PathSyntaxError('Absolute location paths not supported', self.filename, self.lineno)
                elif self.cur_token == '//':
                    steps.append((DESCENDANT_OR_SELF, NodeTest(), []))
                self.next_token()
            axis, nodetest, predicates = self._location_step()
            if not axis:
                axis = CHILD
            steps.append((axis, nodetest, predicates))
            if self.at_end or not self.cur_token.startswith('/'):
                break
        return steps

    def _location_step(self):
        if self.cur_token == '@':
            axis = ATTRIBUTE
            self.next_token()
        elif self.cur_token == '.':
            axis = SELF
        elif self.cur_token == '..':
            raise PathSyntaxError('Unsupported axis "parent"', self.filename, self.lineno)
        elif self.peek_token() == '::':
            axis = Axis.forname(self.cur_token)
            if axis is None:
                raise PathSyntaxError('Unsupport axis "%s"' % axis, self.filename, self.lineno)
            self.next_token()
            self.next_token()
        else:
            axis = None
        nodetest = self._node_test(axis or CHILD)
        predicates = []
        while self.cur_token == '[':
            predicates.append(self._predicate())
        return (axis, nodetest, predicates)

    def _node_test(self, axis=None):
        test = prefix = None
        next_token = self.peek_token()
        if next_token in ('(', '()'):
            test = self._node_type()
        elif next_token == ':':
            prefix = self.cur_token
            self.next_token()
            localname = self.next_token()
            if localname == '*':
                test = QualifiedPrincipalTypeTest(axis, prefix)
            else:
                test = QualifiedNameTest(axis, prefix, localname)
        elif self.cur_token == '*':
            test = PrincipalTypeTest(axis)
        elif self.cur_token == '.':
            test = NodeTest()
        else:
            test = LocalNameTest(axis, self.cur_token)
        if not self.at_end:
            self.next_token()
        return test

    def _node_type(self):
        name = self.cur_token
        self.next_token()
        args = []
        if self.cur_token != '()':
            self.next_token()
            if self.cur_token != ')':
                string = self.cur_token
                if (string[0], string[-1]) in self._QUOTES:
                    string = string[1:-1]
                args.append(string)
        cls = _nodetest_map.get(name)
        if not cls:
            raise PathSyntaxError('%s() not allowed here' % name, self.filename, self.lineno)
        return cls(*args)

    def _predicate(self):
        assert self.cur_token == '['
        self.next_token()
        expr = self._or_expr()
        if self.cur_token != ']':
            raise PathSyntaxError('Expected "]" to close predicate, but found "%s"' % self.cur_token, self.filename, self.lineno)
        if not self.at_end:
            self.next_token()
        return expr

    def _or_expr(self):
        expr = self._and_expr()
        while self.cur_token == 'or':
            self.next_token()
            expr = OrOperator(expr, self._and_expr())
        return expr

    def _and_expr(self):
        expr = self._equality_expr()
        while self.cur_token == 'and':
            self.next_token()
            expr = AndOperator(expr, self._equality_expr())
        return expr

    def _equality_expr(self):
        expr = self._relational_expr()
        while self.cur_token in ('=', '!='):
            op = _operator_map[self.cur_token]
            self.next_token()
            expr = op(expr, self._relational_expr())
        return expr

    def _relational_expr(self):
        expr = self._sub_expr()
        while self.cur_token in ('>', '>=', '<', '>='):
            op = _operator_map[self.cur_token]
            self.next_token()
            expr = op(expr, self._sub_expr())
        return expr

    def _sub_expr(self):
        token = self.cur_token
        if token != '(':
            return self._primary_expr()
        self.next_token()
        expr = self._or_expr()
        if self.cur_token != ')':
            raise PathSyntaxError('Expected ")" to close sub-expression, but found "%s"' % self.cur_token, self.filename, self.lineno)
        self.next_token()
        return expr

    def _primary_expr(self):
        token = self.cur_token
        if len(token) > 1 and (token[0], token[-1]) in self._QUOTES:
            self.next_token()
            return StringLiteral(token[1:-1])
        elif token[0].isdigit() or token[0] == '.':
            self.next_token()
            return NumberLiteral(as_float(token))
        elif token == '$':
            token = self.next_token()
            self.next_token()
            return VariableReference(token)
        elif not self.at_end and self.peek_token().startswith('('):
            return self._function_call()
        else:
            axis = None
            if token == '@':
                axis = ATTRIBUTE
                self.next_token()
            return self._node_test(axis)

    def _function_call(self):
        name = self.cur_token
        if self.next_token() == '()':
            args = []
        else:
            assert self.cur_token == '('
            self.next_token()
            args = [self._or_expr()]
            while self.cur_token == ',':
                self.next_token()
                args.append(self._or_expr())
            if not self.cur_token == ')':
                raise PathSyntaxError('Expected ")" to close function argument list, but found "%s"' % self.cur_token, self.filename, self.lineno)
        self.next_token()
        cls = _function_map.get(name)
        if not cls:
            raise PathSyntaxError('Unsupported function "%s"' % name, self.filename, self.lineno)
        return cls(*args)