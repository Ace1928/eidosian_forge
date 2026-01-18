import itertools
from ..char_classes import (
def _make_ro_variants(tokens):
    variants = []
    for token in tokens:
        upper_token = token.upper()
        upper_char_variants = [_ro_variants.get(c, [c]) for c in upper_token]
        upper_variants = [''.join(x) for x in itertools.product(*upper_char_variants)]
        for variant in upper_variants:
            variants.extend([variant, variant.lower(), variant.title()])
    return sorted(list(set(variants)))