from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def _consume_rule(first_token, tokens):
    """Parse a qualified rule or at-rule.

    Consume just enough of :obj:`tokens` for this rule.

    :type first_token: :term:`component value`
    :param first_token: The first component value of the rule.
    :type tokens: :term:`iterator`
    :param tokens: An iterator yielding :term:`component values`.
    :returns:
        A :class:`~tinycss2.ast.QualifiedRule`,
        :class:`~tinycss2.ast.AtRule`,
        or :class:`~tinycss2.ast.ParseError`.

    """
    if first_token.type == 'at-keyword':
        return _consume_at_rule(first_token, tokens)
    if first_token.type == '{} block':
        prelude = []
        block = first_token
    else:
        prelude = [first_token]
        for token in tokens:
            if token.type == '{} block':
                block = token
                break
            prelude.append(token)
        else:
            return ParseError(prelude[-1].source_line, prelude[-1].source_column, 'invalid', 'EOF reached before {} block for a qualified rule.')
    return QualifiedRule(first_token.source_line, first_token.source_column, prelude, block.content)