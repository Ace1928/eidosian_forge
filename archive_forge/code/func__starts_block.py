import re
import copy
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import iteritems
def _starts_block(token, state):

    def callback(lexer, match, ctx):
        yield (match.start(), token, match.group(0))
        if hasattr(ctx, 'last_indentation'):
            ctx.block_indentation = ctx.last_indentation
        else:
            ctx.block_indentation = ''
        ctx.block_state = state
        ctx.pos = match.end()
    return callback