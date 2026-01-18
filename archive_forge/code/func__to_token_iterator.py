from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def _to_token_iterator(input, skip_comments=False):
    """Iterate component values out of string or component values iterable.

    :type input: :obj:`str` or :term:`iterable`
    :param input: A string or an iterable of :term:`component values`.
    :type skip_comments: :obj:`bool`
    :param skip_comments: If the input is a string, ignore all CSS comments.
    :returns: An iterator yielding :term:`component values`.

    """
    if isinstance(input, str):
        input = parse_component_value_list(input, skip_comments)
    return iter(input)