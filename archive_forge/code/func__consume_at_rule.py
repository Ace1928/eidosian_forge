from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def _consume_at_rule(at_keyword, tokens):
    """Parse an at-rule.

    Consume just enough of :obj:`tokens` for this rule.

    :type at_keyword: :class:`AtKeywordToken`
    :param at_keyword: The at-rule keyword token starting this rule.
    :type tokens: :term:`iterator`
    :param tokens: An iterator yielding :term:`component values`.
    :returns:
        A :class:`~tinycss2.ast.QualifiedRule`,
        or :class:`~tinycss2.ast.ParseError`.

    """
    prelude = []
    content = None
    for token in tokens:
        if token.type == '{} block':
            content = token.content
            break
        elif token == ';':
            break
        prelude.append(token)
    return AtRule(at_keyword.source_line, at_keyword.source_column, at_keyword.value, at_keyword.lower_value, prelude, content)