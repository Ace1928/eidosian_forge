import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class Inform7Lexer(RegexLexer):
    """
    For `Inform 7 <http://inform7.com/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Inform 7'
    aliases = ['inform7', 'i7']
    filenames = ['*.ni', '*.i7x']
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    _dash = Inform6Lexer._dash
    _dquote = Inform6Lexer._dquote
    _newline = Inform6Lexer._newline
    _start = '\\A|(?<=[%s])' % _newline
    tokens = {}
    token_variants = ['+i6t-not-inline', '+i6t-inline', '+i6t-use-option']
    for level in token_variants:
        tokens[level] = {'+i6-root': list(Inform6Lexer.tokens['root']), '+i6t-root': [('[^%s]*' % Inform6Lexer._newline, Comment.Preproc, ('directive', '+p'))], 'root': [('(\\|?\\s)+', Text), ('\\[', Comment.Multiline, '+comment'), ('[%s]' % _dquote, Generic.Heading, ('+main', '+titling', '+titling-string')), default(('+main', '+heading?'))], '+titling-string': [('[^%s]+' % _dquote, Generic.Heading), ('[%s]' % _dquote, Generic.Heading, '#pop')], '+titling': [('\\[', Comment.Multiline, '+comment'), ('[^%s.;:|%s]+' % (_dquote, _newline), Generic.Heading), ('[%s]' % _dquote, Generic.Heading, '+titling-string'), ('[%s]{2}|(?<=[\\s%s])\\|[\\s%s]' % (_newline, _dquote, _dquote), Text, ('#pop', '+heading?')), ('[.;:]|(?<=[\\s%s])\\|' % _dquote, Text, '#pop'), ('[|%s]' % _newline, Generic.Heading)], '+main': [('(?i)[^%s:a\\[(|%s]+' % (_dquote, _newline), Text), ('[%s]' % _dquote, String.Double, '+text'), (':', Text, '+phrase-definition'), ('(?i)\\bas\\b', Text, '+use-option'), ('\\[', Comment.Multiline, '+comment'), ('(\\([%s])(.*?)([%s]\\))' % (_dash, _dash), bygroups(Punctuation, using(this, state=('+i6-root', 'directive'), i6t='+i6t-not-inline'), Punctuation)), ('(%s|(?<=[\\s;:.%s]))\\|\\s|[%s]{2,}' % (_start, _dquote, _newline), Text, '+heading?'), ('(?i)[a(|%s]' % _newline, Text)], '+phrase-definition': [('\\s+', Text), ('\\[', Comment.Multiline, '+comment'), ('(\\([%s])(.*?)([%s]\\))' % (_dash, _dash), bygroups(Punctuation, using(this, state=('+i6-root', 'directive', 'default', 'statements'), i6t='+i6t-inline'), Punctuation), '#pop'), default('#pop')], '+use-option': [('\\s+', Text), ('\\[', Comment.Multiline, '+comment'), ('(\\([%s])(.*?)([%s]\\))' % (_dash, _dash), bygroups(Punctuation, using(this, state=('+i6-root', 'directive'), i6t='+i6t-use-option'), Punctuation), '#pop'), default('#pop')], '+comment': [('[^\\[\\]]+', Comment.Multiline), ('\\[', Comment.Multiline, '#push'), ('\\]', Comment.Multiline, '#pop')], '+text': [('[^\\[%s]+' % _dquote, String.Double), ('\\[.*?\\]', String.Interpol), ('[%s]' % _dquote, String.Double, '#pop')], '+heading?': [('(\\|?\\s)+', Text), ('\\[', Comment.Multiline, '+comment'), ('[%s]{4}\\s+' % _dash, Text, '+documentation-heading'), ('[%s]{1,3}' % _dash, Text), ('(?i)(volume|book|part|chapter|section)\\b[^%s]*' % _newline, Generic.Heading, '#pop'), default('#pop')], '+documentation-heading': [('\\s+', Text), ('\\[', Comment.Multiline, '+comment'), ('(?i)documentation\\s+', Text, '+documentation-heading2'), default('#pop')], '+documentation-heading2': [('\\s+', Text), ('\\[', Comment.Multiline, '+comment'), ('[%s]{4}\\s' % _dash, Text, '+documentation'), default('#pop:2')], '+documentation': [('(?i)(%s)\\s*(chapter|example)\\s*:[^%s]*' % (_start, _newline), Generic.Heading), ('(?i)(%s)\\s*section\\s*:[^%s]*' % (_start, _newline), Generic.Subheading), ('((%s)\\t.*?[%s])+' % (_start, _newline), using(this, state='+main')), ('[^%s\\[]+|[%s\\[]' % (_newline, _newline), Text), ('\\[', Comment.Multiline, '+comment')], '+i6t-not-inline': [('(%s)@c( .*?)?([%s]|\\Z)' % (_start, _newline), Comment.Preproc), ('(%s)@([%s]+|Purpose:)[^%s]*' % (_start, _dash, _newline), Comment.Preproc), ('(%s)@p( .*?)?([%s]|\\Z)' % (_start, _newline), Generic.Heading, '+p')], '+i6t-use-option': [include('+i6t-not-inline'), ('(\\{)(N)(\\})', bygroups(Punctuation, Text, Punctuation))], '+i6t-inline': [('(\\{)(\\S[^}]*)?(\\})', bygroups(Punctuation, using(this, state='+main'), Punctuation))], '+i6t': [('(\\{[%s])(![^}]*)(\\}?)' % _dash, bygroups(Punctuation, Comment.Single, Punctuation)), ('(\\{[%s])(lines)(:)([^}]*)(\\}?)' % _dash, bygroups(Punctuation, Keyword, Punctuation, Text, Punctuation), '+lines'), ('(\\{[%s])([^:}]*)(:?)([^}]*)(\\}?)' % _dash, bygroups(Punctuation, Keyword, Punctuation, Text, Punctuation)), ('(\\(\\+)(.*?)(\\+\\)|\\Z)', bygroups(Punctuation, using(this, state='+main'), Punctuation))], '+p': [('[^@]+', Comment.Preproc), ('(%s)@c( .*?)?([%s]|\\Z)' % (_start, _newline), Comment.Preproc, '#pop'), ('(%s)@([%s]|Purpose:)' % (_start, _dash), Comment.Preproc), ('(%s)@p( .*?)?([%s]|\\Z)' % (_start, _newline), Generic.Heading), ('@', Comment.Preproc)], '+lines': [('(%s)@c( .*?)?([%s]|\\Z)' % (_start, _newline), Comment.Preproc), ('(%s)@([%s]|Purpose:)[^%s]*' % (_start, _dash, _newline), Comment.Preproc), ('(%s)@p( .*?)?([%s]|\\Z)' % (_start, _newline), Generic.Heading, '+p'), ('(%s)@\\w*[ %s]' % (_start, _newline), Keyword), ('![^%s]*' % _newline, Comment.Single), ('(\\{)([%s]endlines)(\\})' % _dash, bygroups(Punctuation, Keyword, Punctuation), '#pop'), ('[^@!{]+?([%s]|\\Z)|.' % _newline, Text)]}
        for token in Inform6Lexer.tokens:
            if token == 'root':
                continue
            tokens[level][token] = list(Inform6Lexer.tokens[token])
            if not token.startswith('_'):
                tokens[level][token][:0] = [include('+i6t'), include(level)]

    def __init__(self, **options):
        level = options.get('i6t', '+i6t-not-inline')
        if level not in self._all_tokens:
            self._tokens = self.__class__.process_tokendef(level)
        else:
            self._tokens = self._all_tokens[level]
        RegexLexer.__init__(self, **options)