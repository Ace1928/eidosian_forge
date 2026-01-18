import re
from pygments.lexer import RegexLexer, include, bygroups, using, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.html import HtmlLexer
from pygments.lexers import _stan_builtins
class ModelicaLexer(RegexLexer):
    """
    For `Modelica <http://www.modelica.org/>`_ source code.

    .. versionadded:: 1.1
    """
    name = 'Modelica'
    aliases = ['modelica']
    filenames = ['*.mo']
    mimetypes = ['text/x-modelica']
    flags = re.DOTALL | re.MULTILINE
    _name = "(?:'(?:[^\\\\']|\\\\.)+'|[a-zA-Z_]\\w*)"
    tokens = {'whitespace': [(u'[\\s\ufeff]+', Text), ('//[^\\n]*\\n?', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline)], 'root': [include('whitespace'), ('"', String.Double, 'string'), ('[()\\[\\]{},;]+', Punctuation), ('\\.?[*^/+-]|\\.|<>|[<>:=]=?', Operator), ('\\d+(\\.?\\d*[eE][-+]?\\d+|\\.\\d*)', Number.Float), ('\\d+', Number.Integer), ('(abs|acos|actualStream|array|asin|assert|AssertionLevel|atan|atan2|backSample|Boolean|cardinality|cat|ceil|change|Clock|Connections|cos|cosh|cross|delay|diagonal|div|edge|exp|ExternalObject|fill|floor|getInstanceName|hold|homotopy|identity|inStream|integer|Integer|interval|inverse|isPresent|linspace|log|log10|matrix|max|min|mod|ndims|noClock|noEvent|ones|outerProduct|pre|previous|product|Real|reinit|rem|rooted|sample|scalar|semiLinear|shiftSample|sign|sin|sinh|size|skew|smooth|spatialDistribution|sqrt|StateSelect|String|subSample|sum|superSample|symmetric|tan|tanh|terminal|terminate|time|transpose|vector|zeros)\\b', Name.Builtin), ('(algorithm|annotation|break|connect|constant|constrainedby|der|discrete|each|else|elseif|elsewhen|encapsulated|enumeration|equation|exit|expandable|extends|external|final|flow|for|if|import|impure|in|initial|inner|input|loop|nondiscrete|outer|output|parameter|partial|protected|public|pure|redeclare|replaceable|return|stream|then|when|while)\\b', Keyword.Reserved), ('(and|not|or)\\b', Operator.Word), ('(block|class|connector|end|function|model|operator|package|record|type)\\b', Keyword.Reserved, 'class'), ('(false|true)\\b', Keyword.Constant), ('within\\b', Keyword.Reserved, 'package-prefix'), (_name, Name)], 'class': [include('whitespace'), ('(function|record)\\b', Keyword.Reserved), ('(if|for|when|while)\\b', Keyword.Reserved, '#pop'), (_name, Name.Class, '#pop'), default('#pop')], 'package-prefix': [include('whitespace'), (_name, Name.Namespace, '#pop'), default('#pop')], 'string': [('"', String.Double, '#pop'), ('\\\\[\\\'"?\\\\abfnrtv]', String.Escape), ('(?i)<\\s*html\\s*>([^\\\\"]|\\\\.)+?(<\\s*/\\s*html\\s*>|(?="))', using(HtmlLexer)), ('<|\\\\?[^"\\\\<]+', String.Double)]}