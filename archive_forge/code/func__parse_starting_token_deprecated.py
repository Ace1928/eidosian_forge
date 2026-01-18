import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _parse_starting_token_deprecated(self):
    """
        This handles parsing of old style starting tokens, and attempts to
        coerce them into the new style.
        """
    log.debug('Attempting to fall back to old starting token parser. For token: %s' % self._starting_token)
    if self._starting_token is None:
        return None
    parts = self._starting_token.split('___')
    next_token = []
    index = 0
    if len(parts) == len(self._input_token) + 1:
        try:
            index = int(parts.pop())
        except ValueError:
            parts = [self._starting_token]
    for part in parts:
        if part == 'None':
            next_token.append(None)
        else:
            next_token.append(part)
    return (self._convert_deprecated_starting_token(next_token), index)