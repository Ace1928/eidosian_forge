from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class ConditionalGroupNode(ArgGroupNode):

    @classmethod
    def parse(cls, ctx, tokens, breakstack):
        """
    Parser for the commands that take conditional arguments. Similar to the
    standard parser but it understands parentheses and can generate
    parenthentical groups::

        while(CONDITION1 AND (CONDITION2 OR CONDITION3)
              OR (CONDITION3 AND (CONDITION4 AND CONDITION5)
              OR CONDITION6)
    """
        kwargs = {'AND': cls.parse, 'OR': cls.parse}
        flags = list(CONDITIONAL_FLAGS)
        tree = cls()
        while tokens and tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        flags = [flag.upper() for flag in flags]
        breaker = KwargBreaker(list(kwargs.keys()))
        child_breakstack = breakstack + [breaker]
        while tokens:
            if should_break(tokens[0], breakstack):
                break
            if tokens[0].type in WHITESPACE_TOKENS:
                tree.children.append(tokens.pop(0))
                continue
            if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
                child = CommentNode()
                tree.children.append(child)
                child.children.append(tokens.pop(0))
                continue
            if tokens[0].type in (lex.TokenType.FORMAT_OFF, lex.TokenType.FORMAT_ON):
                tree.children.append(OnOffNode.consume(ctx, tokens))
                continue
            if tokens[0].type == lex.TokenType.LEFT_PAREN:
                with ctx.pusharg(tree):
                    subtree = ParenGroupNode.parse(ctx, tokens, breakstack)
                tree.children.append(subtree)
                continue
            ntokens = len(tokens)
            word = get_normalized_kwarg(tokens[0])
            if word in kwargs:
                with ctx.pusharg(tree):
                    subtree = KeywordGroupNode.parse(ctx, tokens, word, kwargs[word], child_breakstack)
                assert len(tokens) < ntokens, 'parsed an empty subtree'
                tree.children.append(subtree)
                continue
            with ctx.pusharg(tree):
                child = PositionalGroupNode.parse(ctx, tokens, '+', flags, child_breakstack)
            tree.children.append(child)
        return tree