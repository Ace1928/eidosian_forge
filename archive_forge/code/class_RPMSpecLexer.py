import re
from pygments.lexer import RegexLexer, include, bygroups, using, this, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RPMSpecLexer(RegexLexer):
    """
    For RPM ``.spec`` files.

    .. versionadded:: 1.6
    """
    name = 'RPMSpec'
    aliases = ['spec']
    filenames = ['*.spec']
    mimetypes = ['text/x-rpm-spec']
    _directives = '(?:package|prep|build|install|clean|check|pre[a-z]*|post[a-z]*|trigger[a-z]*|files)'
    tokens = {'root': [('#.*\\n', Comment), include('basic')], 'description': [('^(%' + _directives + ')(.*)$', bygroups(Name.Decorator, Text), '#pop'), ('\\n', Text), ('.', Text)], 'changelog': [('\\*.*\\n', Generic.Subheading), ('^(%' + _directives + ')(.*)$', bygroups(Name.Decorator, Text), '#pop'), ('\\n', Text), ('.', Text)], 'string': [('"', String.Double, '#pop'), ('\\\\([\\\\abfnrtv"\\\']|x[a-fA-F0-9]{2,4}|[0-7]{1,3})', String.Escape), include('interpol'), ('.', String.Double)], 'basic': [include('macro'), ('(?i)^(Name|Version|Release|Epoch|Summary|Group|License|Packager|Vendor|Icon|URL|Distribution|Prefix|Patch[0-9]*|Source[0-9]*|Requires\\(?[a-z]*\\)?|[a-z]+Req|Obsoletes|Suggests|Provides|Conflicts|Build[a-z]+|[a-z]+Arch|Auto[a-z]+)(:)(.*)$', bygroups(Generic.Heading, Punctuation, using(this))), ('^%description', Name.Decorator, 'description'), ('^%changelog', Name.Decorator, 'changelog'), ('^(%' + _directives + ')(.*)$', bygroups(Name.Decorator, Text)), ('%(attr|defattr|dir|doc(?:dir)?|setup|config(?:ure)?|make(?:install)|ghost|patch[0-9]+|find_lang|exclude|verify)', Keyword), include('interpol'), ("'.*?'", String.Single), ('"', String.Double, 'string'), ('.', Text)], 'macro': [('%define.*\\n', Comment.Preproc), ('%\\{\\!\\?.*%define.*\\}', Comment.Preproc), ('(%(?:if(?:n?arch)?|else(?:if)?|endif))(.*)$', bygroups(Comment.Preproc, Text))], 'interpol': [('%\\{?__[a-z_]+\\}?', Name.Function), ('%\\{?_([a-z_]+dir|[a-z_]+path|prefix)\\}?', Keyword.Pseudo), ('%\\{\\?\\w+\\}', Name.Variable), ('\\$\\{?RPM_[A-Z0-9_]+\\}?', Name.Variable.Global), ('%\\{[a-zA-Z]\\w+\\}', Keyword.Constant)]}