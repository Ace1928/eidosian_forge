import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class PropertiesLexer(RegexLexer):
    """
    Lexer for configuration files in Java's properties format.

    Note: trailing whitespace counts as part of the value as per spec

    .. versionadded:: 1.4
    """
    name = 'Properties'
    aliases = ['properties', 'jproperties']
    filenames = ['*.properties']
    mimetypes = ['text/x-java-properties']
    tokens = {'root': [('^(\\w+)([ \\t])(\\w+\\s*)$', bygroups(Name.Attribute, Text, String)), ('^\\w+(\\\\[ \\t]\\w*)*$', Name.Attribute), ('(^ *)([#!].*)', bygroups(Text, Comment)), ('(^ *)((?:;|//).*)', bygroups(Text, Comment)), ('(.*?)([ \\t]*)([=:])([ \\t]*)(.*(?:(?<=\\\\)\\n.*)*)', bygroups(Name.Attribute, Text, Operator, Text, String)), ('\\s', Text)]}