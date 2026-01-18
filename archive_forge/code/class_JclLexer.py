import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class JclLexer(RegexLexer):
    """
    `Job Control Language (JCL)
    <http://publibz.boulder.ibm.com/cgi-bin/bookmgr_OS390/BOOKS/IEA2B570/CCONTENTS>`_
    is a scripting language used on mainframe platforms to instruct the system
    on how to run a batch job or start a subsystem. It is somewhat
    comparable to MS DOS batch and Unix shell scripts.

    .. versionadded:: 2.1
    """
    name = 'JCL'
    aliases = ['jcl']
    filenames = ['*.jcl']
    mimetypes = ['text/x-jcl']
    flags = re.IGNORECASE
    tokens = {'root': [('//\\*.*\\n', Comment.Single), ('//', Keyword.Pseudo, 'statement'), ('/\\*', Keyword.Pseudo, 'jes2_statement'), ('.*\\n', Other)], 'statement': [('\\s*\\n', Whitespace, '#pop'), ('([a-z]\\w*)(\\s+)(exec|job)(\\s*)', bygroups(Name.Label, Whitespace, Keyword.Reserved, Whitespace), 'option'), ('[a-z]\\w*', Name.Variable, 'statement_command'), ('\\s+', Whitespace, 'statement_command')], 'statement_command': [('\\s+(command|cntl|dd|endctl|endif|else|include|jcllib|output|pend|proc|set|then|xmit)\\s+', Keyword.Reserved, 'option'), include('option')], 'jes2_statement': [('\\s*\\n', Whitespace, '#pop'), ('\\$', Keyword, 'option'), ('\\b(jobparam|message|netacct|notify|output|priority|route|setup|signoff|xeq|xmit)\\b', Keyword, 'option')], 'option': [('\\*', Name.Builtin), ('[\\[\\](){}<>;,]', Punctuation), ('[-+*/=&%]', Operator), ('[a-z_]\\w*', Name), ('\\d+\\.\\d*', Number.Float), ('\\.\\d+', Number.Float), ('\\d+', Number.Integer), ("'", String, 'option_string'), ('[ \\t]+', Whitespace, 'option_comment'), ('\\.', Punctuation)], 'option_string': [('(\\n)(//)', bygroups(Text, Keyword.Pseudo)), ("''", String), ("[^']", String), ("'", String, '#pop')], 'option_comment': [('.+', Comment.Single)]}
    _JOB_HEADER_PATTERN = re.compile('^//[a-z#$@][a-z0-9#$@]{0,7}\\s+job(\\s+.*)?$', re.IGNORECASE)

    def analyse_text(text):
        """
        Recognize JCL job by header.
        """
        result = 0.0
        lines = text.split('\n')
        if len(lines) > 0:
            if JclLexer._JOB_HEADER_PATTERN.match(lines[0]):
                result = 1.0
        assert 0.0 <= result <= 1.0
        return result