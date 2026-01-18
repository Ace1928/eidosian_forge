import re
import copy
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import iteritems
def _indentation(lexer, match, ctx):
    indentation = match.group(0)
    yield (match.start(), Text, indentation)
    ctx.last_indentation = indentation
    ctx.pos = match.end()
    if hasattr(ctx, 'block_state') and ctx.block_state and indentation.startswith(ctx.block_indentation) and (indentation != ctx.block_indentation):
        ctx.stack.append(ctx.block_state)
    else:
        ctx.block_state = None
        ctx.block_indentation = None
        ctx.stack.append('content')