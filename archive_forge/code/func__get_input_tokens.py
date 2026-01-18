import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _get_input_tokens(self, config):
    input_token = self._pagination_cfg['input_token']
    if not isinstance(input_token, list):
        input_token = [input_token]
    return input_token