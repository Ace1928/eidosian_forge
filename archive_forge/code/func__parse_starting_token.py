import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _parse_starting_token(self):
    if self._starting_token is None:
        return None
    next_token = self._starting_token
    try:
        next_token = self._token_decoder.decode(next_token)
        index = 0
        if 'boto_truncate_amount' in next_token:
            index = next_token.get('boto_truncate_amount')
            del next_token['boto_truncate_amount']
    except (ValueError, TypeError):
        next_token, index = self._parse_starting_token_deprecated()
    return (next_token, index)