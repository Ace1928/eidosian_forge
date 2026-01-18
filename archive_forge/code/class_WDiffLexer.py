import re
from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, Generic, \
class WDiffLexer(RegexLexer):
    """
    A `wdiff <https://www.gnu.org/software/wdiff/>`_ lexer.

    Note that:

    * only to normal output (without option like -l).
    * if target files of wdiff contain "[-", "-]", "{+", "+}",
      especially they are unbalanced, this lexer will get confusing.

    .. versionadded:: 2.2
    """
    name = 'WDiff'
    aliases = ['wdiff']
    filenames = ['*.wdiff']
    mimetypes = []
    flags = re.MULTILINE | re.DOTALL
    ins_op = '\\{\\+'
    ins_cl = '\\+\\}'
    del_op = '\\[\\-'
    del_cl = '\\-\\]'
    normal = '[^{}[\\]+-]+'
    tokens = {'root': [(ins_op, Generic.Inserted, 'inserted'), (del_op, Generic.Deleted, 'deleted'), (normal, Text), ('.', Text)], 'inserted': [(ins_op, Generic.Inserted, '#push'), (del_op, Generic.Inserted, '#push'), (del_cl, Generic.Inserted, '#pop'), (ins_cl, Generic.Inserted, '#pop'), (normal, Generic.Inserted), ('.', Generic.Inserted)], 'deleted': [(del_op, Generic.Deleted, '#push'), (ins_op, Generic.Deleted, '#push'), (ins_cl, Generic.Deleted, '#pop'), (del_cl, Generic.Deleted, '#pop'), (normal, Generic.Deleted), ('.', Generic.Deleted)]}