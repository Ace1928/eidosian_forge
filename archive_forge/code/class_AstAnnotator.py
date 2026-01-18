from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys
from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator
class AstAnnotator(BaseVisitor):

    def __init__(self, source):
        super(AstAnnotator, self).__init__()
        self.tokens = token_generator.TokenGenerator(source)

    def visit(self, node):
        try:
            fmt.set(node, 'indent', self._indent)
            fmt.set(node, 'indent_diff', self._indent_diff)
            super(AstAnnotator, self).visit(node)
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise AnnotationError(e)

    def indented(self, node, children_attr):
        """Generator which annotates child nodes with their indentation level."""
        children = getattr(node, children_attr)
        cur_loc = self.tokens._loc
        next_loc = self.tokens.peek_non_whitespace().start
        if cur_loc[0] == next_loc[0]:
            indent_diff = self._indent_diff
            self._indent_diff = None
            for child in children:
                yield child
            self._indent_diff = indent_diff
            return
        prev_indent = self._indent
        prev_indent_diff = self._indent_diff
        indent_token = self.tokens.peek_conditional(lambda t: t.type == token_generator.TOKENS.INDENT)
        new_indent = indent_token.src
        new_diff = _get_indent_diff(prev_indent, new_indent)
        if not new_diff:
            new_diff = ' ' * 4
            print('Indent detection failed (line %d); inner indentation level is not more than the outer indentation.' % cur_loc[0], file=sys.stderr)
        self._indent = new_indent
        self._indent_diff = new_diff
        for child in children:
            yield child
        fmt.set(node, 'block_suffix_%s' % children_attr, self.tokens.block_whitespace(self._indent))
        self._indent = prev_indent
        self._indent_diff = prev_indent_diff

    @expression
    def visit_Num(self, node):
        """Annotate a Num node with the exact number format."""
        token_number_type = token_generator.TOKENS.NUMBER
        contentargs = [lambda: self.tokens.next_of_type(token_number_type).src]
        if self.tokens.peek().src == '-':
            contentargs.insert(0, '-')
        self.attr(node, 'content', contentargs, deps=('n',), default=str(node.n))

    @expression
    def visit_Str(self, node):
        """Annotate a Str node with the exact string format."""
        self.attr(node, 'content', [self.tokens.str], deps=('s',), default=node.s)

    @expression
    def visit_JoinedStr(self, node):
        """Annotate a JoinedStr node with the fstr formatting metadata."""
        fstr_iter = self.tokens.fstr()()
        res = ''
        values = (v for v in node.values if isinstance(v, ast.FormattedValue))
        while True:
            res_part, tg = next(fstr_iter)
            res += res_part
            if tg is None:
                break
            prev_tokens = self.tokens
            self.tokens = tg
            self.visit(next(values))
            self.tokens = prev_tokens
        self.attr(node, 'content', [lambda: res], default=res)

    @expression
    def visit_Bytes(self, node):
        """Annotate a Bytes node with the exact string format."""
        self.attr(node, 'content', [self.tokens.str], deps=('s',), default=node.s)

    @space_around
    def visit_Ellipsis(self, node):
        if self.tokens.peek().src == '...':
            self.token('...')
        else:
            for i in range(3):
                self.token('.')

    def check_is_elif(self, node):
        """Return True iff the If node is an `elif` in the source."""
        next_tok = self.tokens.next_name()
        return isinstance(node, ast.If) and next_tok.src == 'elif'

    def check_is_continued_try(self, node):
        """Return True iff the TryExcept node is a continued `try` in the source."""
        return isinstance(node, ast.TryExcept) and self.tokens.peek_non_whitespace().src != 'try'

    def check_is_continued_with(self, node):
        """Return True iff the With node is a continued `with` in the source."""
        return isinstance(node, ast.With) and self.tokens.peek().src == ','

    def check_slice_includes_step(self, node):
        """Helper function for Slice node to determine whether to visit its step."""
        return self.tokens.peek_non_whitespace().src not in '],'

    def ws(self, max_lines=None, semicolon=False, comment=True):
        """Parse some whitespace from the source tokens and return it."""
        next_token = self.tokens.peek()
        if semicolon and next_token and (next_token.src == ';'):
            result = self.tokens.whitespace() + self.token(';')
            next_token = self.tokens.peek()
            if next_token.type in (token_generator.TOKENS.NL, token_generator.TOKENS.NEWLINE):
                result += self.tokens.whitespace(max_lines=1)
            return result
        return self.tokens.whitespace(max_lines=max_lines, comment=comment)

    def dots(self, num_dots):
        """Parse a number of dots."""

        def _parse_dots():
            return self.tokens.dots(num_dots)
        return _parse_dots

    def block_suffix(self, node, indent_level):
        fmt.set(node, 'suffix', self.tokens.block_whitespace(indent_level))

    def token(self, token_val):
        """Parse a single token with exactly the given value."""
        token = self.tokens.next()
        if token.src != token_val:
            raise AnnotationError('Expected %r but found %r\nline %d: %s' % (token_val, token.src, token.start[0], token.line))
        if token.src in '({[':
            self.tokens.hint_open()
        elif token.src in ')}]':
            self.tokens.hint_closed()
        return token.src

    def optional_token(self, node, attr_name, token_val, allow_whitespace_prefix=False, default=False):
        """Try to parse a token and attach it to the node."""
        del default
        fmt.append(node, attr_name, '')
        token = self.tokens.peek_non_whitespace() if allow_whitespace_prefix else self.tokens.peek()
        if token and token.src == token_val:
            parsed = ''
            if allow_whitespace_prefix:
                parsed += self.ws()
            fmt.append(node, attr_name, parsed + self.tokens.next().src + self.ws())

    def one_of_symbols(self, *symbols):
        """Account for one of the given symbols."""

        def _one_of_symbols():
            next_token = self.tokens.next()
            found = next((s for s in symbols if s == next_token.src), None)
            if found is None:
                raise AnnotationError('Expected one of: %r, but found: %r' % (symbols, next_token.src))
            return found
        return _one_of_symbols

    def attr(self, node, attr_name, attr_vals, deps=None, default=None):
        """Parses some source and sets an attribute on the given node.

    Stores some arbitrary formatting information on the node. This takes a list
    attr_vals which tell what parts of the source to parse. The result of each
    function is concatenated onto the formatting data, and strings in this list
    are a shorthand to look for an exactly matching token.

    For example:
      self.attr(node, 'foo', ['(', self.ws, 'Hello, world!', self.ws, ')'],
                deps=('s',), default=node.s)

    is a rudimentary way to parse a parenthesized string. After running this,
    the matching source code for this node will be stored in its formatting
    dict under the key 'foo'. The result might be `(
  'Hello, world!'
)`.

    This also keeps track of the current value of each of the dependencies.
    In the above example, we would have looked for the string 'Hello, world!'
    because that's the value of node.s, however, when we print this back, we
    want to know if the value of node.s has changed since this time. If any of
    the dependent values has changed, the default would be used instead.

    Arguments:
      node: (ast.AST) An AST node to attach formatting information to.
      attr_name: (string) Name to store the formatting information under.
      attr_vals: (list of functions/strings) Each item is either a function
        that parses some source and return a string OR a string to match
        exactly (as a token).
      deps: (optional, set of strings) Attributes of the node which attr_vals
        depends on.
      default: (string) Unused here.
    """
        del default
        if deps:
            for dep in deps:
                fmt.set(node, dep + '__src', getattr(node, dep, None))
        attr_parts = []
        for attr_val in attr_vals:
            if isinstance(attr_val, six.string_types):
                attr_parts.append(self.token(attr_val))
            else:
                attr_parts.append(attr_val())
        fmt.set(node, attr_name, ''.join(attr_parts))

    def scope(self, node, attr=None, trailing_comma=False, default_parens=False):
        """Return a context manager to handle a parenthesized scope.

    Arguments:
      node: (ast.AST) Node to store the scope prefix and suffix on.
      attr: (string, optional) Attribute of the node contained in the scope, if
        any. For example, as `None`, the scope would wrap the entire node, but
        as 'bases', the scope might wrap only the bases of a class.
      trailing_comma: (boolean) If True, allow a trailing comma at the end.
      default_parens: (boolean) If True and no formatting information is
        present, the scope would be assumed to be parenthesized.
    """
        del default_parens
        return self.tokens.scope(node, attr=attr, trailing_comma=trailing_comma)

    def _optional_token(self, token_type, token_val):
        token = self.tokens.peek()
        if not token or token.type != token_type or token.src != token_val:
            return ''
        else:
            self.tokens.next()
            return token.src + self.ws()