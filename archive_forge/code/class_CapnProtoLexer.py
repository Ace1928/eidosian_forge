import re
from pygments.lexer import RegexLexer, default
from pygments.token import Text, Comment, Keyword, Name, Literal
class CapnProtoLexer(RegexLexer):
    """
    For `Cap'n Proto <https://capnproto.org>`_ source.

    .. versionadded:: 2.2
    """
    name = "Cap'n Proto"
    filenames = ['*.capnp']
    aliases = ['capnp']
    flags = re.MULTILINE | re.UNICODE
    tokens = {'root': [('#.*?$', Comment.Single), ('@[0-9a-zA-Z]*', Name.Decorator), ('=', Literal, 'expression'), (':', Name.Class, 'type'), ('\\$', Name.Attribute, 'annotation'), ('(struct|enum|interface|union|import|using|const|annotation|extends|in|of|on|as|with|from|fixed)\\b', Keyword), ('[\\w.]+', Name), ('[^#@=:$\\w]+', Text)], 'type': [('[^][=;,(){}$]+', Name.Class), ('[[(]', Name.Class, 'parentype'), default('#pop')], 'parentype': [('[^][;()]+', Name.Class), ('[[(]', Name.Class, '#push'), ('[])]', Name.Class, '#pop'), default('#pop')], 'expression': [('[^][;,(){}$]+', Literal), ('[[(]', Literal, 'parenexp'), default('#pop')], 'parenexp': [('[^][;()]+', Literal), ('[[(]', Literal, '#push'), ('[])]', Literal, '#pop'), default('#pop')], 'annotation': [('[^][;,(){}=:]+', Name.Attribute), ('[[(]', Name.Attribute, 'annexp'), default('#pop')], 'annexp': [('[^][;()]+', Name.Attribute), ('[[(]', Name.Attribute, '#push'), ('[])]', Name.Attribute, '#pop'), default('#pop')]}