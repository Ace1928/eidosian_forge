from pygments.lexer import RegexLexer, bygroups, do_insertions, include, \
from pygments.token import Comment, Error, Keyword, Name, Number, Operator, \
from pygments.util import ClassNotFound, get_bool_opt
Adds syntax from another languages inside annotated strings

        match args:
            1:open_string,
            2:exclamation_mark,
            3:lang_name,
            4:space_or_newline,
            5:code,
            6:close_string
        