from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def _consume_declaration_in_list(first_token, tokens):
    """Like :func:`_parse_declaration`, but stop at the first ``;``."""
    other_declaration_tokens = []
    for token in tokens:
        if token == ';':
            break
        other_declaration_tokens.append(token)
    return _parse_declaration(first_token, iter(other_declaration_tokens))