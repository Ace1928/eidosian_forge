import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
class ShenLexer(RegexLexer):
    """
    Lexer for `Shen <http://shenlanguage.org/>`_ source code.

    .. versionadded:: 2.1
    """
    name = 'Shen'
    aliases = ['shen']
    filenames = ['*.shen']
    mimetypes = ['text/x-shen', 'application/x-shen']
    DECLARATIONS = ('datatype', 'define', 'defmacro', 'defprolog', 'defcc', 'synonyms', 'declare', 'package', 'type', 'function')
    SPECIAL_FORMS = ('lambda', 'get', 'let', 'if', 'cases', 'cond', 'put', 'time', 'freeze', 'value', 'load', '$', 'protect', 'or', 'and', 'not', 'do', 'output', 'prolog?', 'trap-error', 'error', 'make-string', '/.', 'set', '@p', '@s', '@v')
    BUILTINS = ('==', '=', '*', '+', '-', '/', '<', '>', '>=', '<=', '<-address', '<-vector', 'abort', 'absvector', 'absvector?', 'address->', 'adjoin', 'append', 'arity', 'assoc', 'bind', 'boolean?', 'bound?', 'call', 'cd', 'close', 'cn', 'compile', 'concat', 'cons', 'cons?', 'cut', 'destroy', 'difference', 'element?', 'empty?', 'enable-type-theory', 'error-to-string', 'eval', 'eval-kl', 'exception', 'explode', 'external', 'fail', 'fail-if', 'file', 'findall', 'fix', 'fst', 'fwhen', 'gensym', 'get-time', 'hash', 'hd', 'hdstr', 'hdv', 'head', 'identical', 'implementation', 'in', 'include', 'include-all-but', 'inferences', 'input', 'input+', 'integer?', 'intern', 'intersection', 'is', 'kill', 'language', 'length', 'limit', 'lineread', 'loaded', 'macro', 'macroexpand', 'map', 'mapcan', 'maxinferences', 'mode', 'n->string', 'nl', 'nth', 'null', 'number?', 'occurrences', 'occurs-check', 'open', 'os', 'out', 'port', 'porters', 'pos', 'pr', 'preclude', 'preclude-all-but', 'print', 'profile', 'profile-results', 'ps', 'quit', 'read', 'read+', 'read-byte', 'read-file', 'read-file-as-bytelist', 'read-file-as-string', 'read-from-string', 'release', 'remove', 'return', 'reverse', 'run', 'save', 'set', 'simple-error', 'snd', 'specialise', 'spy', 'step', 'stinput', 'stoutput', 'str', 'string->n', 'string->symbol', 'string?', 'subst', 'symbol?', 'systemf', 'tail', 'tc', 'tc?', 'thaw', 'tl', 'tlstr', 'tlv', 'track', 'tuple?', 'undefmacro', 'unify', 'unify!', 'union', 'unprofile', 'unspecialise', 'untrack', 'variable?', 'vector', 'vector->', 'vector?', 'verified', 'version', 'warn', 'when', 'write-byte', 'write-to-file', 'y-or-n?')
    BUILTINS_ANYWHERE = ('where', 'skip', '>>', '_', '!', '<e>', '<!>')
    MAPPINGS = dict(((s, Keyword) for s in DECLARATIONS))
    MAPPINGS.update(((s, Name.Builtin) for s in BUILTINS))
    MAPPINGS.update(((s, Keyword) for s in SPECIAL_FORMS))
    valid_symbol_chars = "[\\w!$%*+,<=>?/.\\'@&#:-]"
    valid_name = '%s+' % valid_symbol_chars
    symbol_name = "[a-z!$%%*+,<=>?/.\\'@&#_-]%s*" % valid_symbol_chars
    variable = '[A-Z]%s*' % valid_symbol_chars
    tokens = {'string': [('"', String, '#pop'), ('c#\\d{1,3};', String.Escape), ('~[ARS%]', String.Interpol), ('(?s).', String)], 'root': [('(?s)\\\\\\*.*?\\*\\\\', Comment.Multiline), ('\\\\\\\\.*', Comment.Single), ('\\s+', Text), ('_{5,}', Punctuation), ('={5,}', Punctuation), ('(;|:=|\\||--?>|<--?)', Punctuation), ('(:-|:|\\{|\\})', Literal), ('[+-]*\\d*\\.\\d+(e[+-]?\\d+)?', Number.Float), ('[+-]*\\d+', Number.Integer), ('"', String, 'string'), (variable, Name.Variable), ('(true|false|<>|\\[\\])', Keyword.Pseudo), (symbol_name, Literal), ('(\\[|\\]|\\(|\\))', Punctuation)]}

    def get_tokens_unprocessed(self, text):
        tokens = RegexLexer.get_tokens_unprocessed(self, text)
        tokens = self._process_symbols(tokens)
        tokens = self._process_declarations(tokens)
        return tokens

    def _relevant(self, token):
        return token not in (Text, Comment.Single, Comment.Multiline)

    def _process_declarations(self, tokens):
        opening_paren = False
        for index, token, value in tokens:
            yield (index, token, value)
            if self._relevant(token):
                if opening_paren and token == Keyword and (value in self.DECLARATIONS):
                    declaration = value
                    for index, token, value in self._process_declaration(declaration, tokens):
                        yield (index, token, value)
                opening_paren = value == '(' and token == Punctuation

    def _process_symbols(self, tokens):
        opening_paren = False
        for index, token, value in tokens:
            if opening_paren and token in (Literal, Name.Variable):
                token = self.MAPPINGS.get(value, Name.Function)
            elif token == Literal and value in self.BUILTINS_ANYWHERE:
                token = Name.Builtin
            opening_paren = value == '(' and token == Punctuation
            yield (index, token, value)

    def _process_declaration(self, declaration, tokens):
        for index, token, value in tokens:
            if self._relevant(token):
                break
            yield (index, token, value)
        if declaration == 'datatype':
            prev_was_colon = False
            token = Keyword.Type if token == Literal else token
            yield (index, token, value)
            for index, token, value in tokens:
                if prev_was_colon and token == Literal:
                    token = Keyword.Type
                yield (index, token, value)
                if self._relevant(token):
                    prev_was_colon = token == Literal and value == ':'
        elif declaration == 'package':
            token = Name.Namespace if token == Literal else token
            yield (index, token, value)
        elif declaration == 'define':
            token = Name.Function if token == Literal else token
            yield (index, token, value)
            for index, token, value in tokens:
                if self._relevant(token):
                    break
                yield (index, token, value)
            if value == '{' and token == Literal:
                yield (index, Punctuation, value)
                for index, token, value in self._process_signature(tokens):
                    yield (index, token, value)
            else:
                yield (index, token, value)
        else:
            token = Name.Function if token == Literal else token
            yield (index, token, value)
        raise StopIteration

    def _process_signature(self, tokens):
        for index, token, value in tokens:
            if token == Literal and value == '}':
                yield (index, Punctuation, value)
                raise StopIteration
            elif token in (Literal, Name.Function):
                token = Name.Variable if value.istitle() else Keyword.Type
            yield (index, token, value)