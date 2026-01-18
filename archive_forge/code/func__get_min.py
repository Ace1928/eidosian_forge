import re
import numpy as np
def _get_min(self, tokens):
    next = tokens.pop(0)
    if not next.type == 'DOT':
        raise SyntaxError()
    next = tokens.pop(0)
    return next.value