import re
from collections import OrderedDict
from typing import Any, Optional
def _group_pairs(tokens: list) -> None:
    i = 0
    while i < len(tokens) - 2:
        if tokens[i][0] == 'token' and tokens[i + 1][0] == 'equals' and (tokens[i + 2][0] == 'token'):
            tokens[i:i + 3] = [('pair', (tokens[i][1], tokens[i + 2][1]))]
        i += 1