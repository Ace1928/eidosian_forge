import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer, LassoLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.php import PhpLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
from pygments.lexers.jvm import JavaLexer, TeaLangLexer
from pygments.lexers.data import YamlLexer
from pygments.lexer import Lexer, DelegatingLexer, RegexLexer, bygroups, \
from pygments.token import Error, Punctuation, Whitespace, \
from pygments.util import html_doctype_matches, looks_like_xml
class MyghtyLexer(RegexLexer):
    """
    Generic `myghty templates`_ lexer. Code that isn't Myghty
    markup is yielded as `Token.Other`.

    .. versionadded:: 0.6

    .. _myghty templates: http://www.myghty.org/
    """
    name = 'Myghty'
    aliases = ['myghty']
    filenames = ['*.myt', 'autodelegate']
    mimetypes = ['application/x-myghty']
    tokens = {'root': [('\\s+', Text), ('(<%(?:def|method))(\\s*)(.*?)(>)(.*?)(</%\\2\\s*>)(?s)', bygroups(Name.Tag, Text, Name.Function, Name.Tag, using(this), Name.Tag)), ('(<%\\w+)(.*?)(>)(.*?)(</%\\2\\s*>)(?s)', bygroups(Name.Tag, Name.Function, Name.Tag, using(PythonLexer), Name.Tag)), ('(<&[^|])(.*?)(,.*?)?(&>)', bygroups(Name.Tag, Name.Function, using(PythonLexer), Name.Tag)), ('(<&\\|)(.*?)(,.*?)?(&>)(?s)', bygroups(Name.Tag, Name.Function, using(PythonLexer), Name.Tag)), ('</&>', Name.Tag), ('(<%!?)(.*?)(%>)(?s)', bygroups(Name.Tag, using(PythonLexer), Name.Tag)), ('(?<=^)#[^\\n]*(\\n|\\Z)', Comment), ('(?<=^)(%)([^\\n]*)(\\n|\\Z)', bygroups(Name.Tag, using(PythonLexer), Other)), ("(?sx)\n                 (.+?)               # anything, followed by:\n                 (?:\n                  (?<=\\n)(?=[%#]) |  # an eval or comment line\n                  (?=</?[%&]) |      # a substitution or block or\n                                     # call start or end\n                                     # - don't consume\n                  (\\\\\\n) |           # an escaped newline\n                  \\Z                 # end of string\n                 )", bygroups(Other, Operator))]}