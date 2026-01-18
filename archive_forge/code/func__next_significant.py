from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def _next_significant(tokens):
    """Return the next significant (neither whitespace or comment) token.

    :type tokens: :term:`iterator`
    :param tokens: An iterator yielding :term:`component values`.
    :returns: A :term:`component value`, or :obj:`None`.

    """
    for token in tokens:
        if token.type not in ('whitespace', 'comment'):
            return token