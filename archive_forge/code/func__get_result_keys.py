import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _get_result_keys(self, config):
    result_key = config.get('result_key')
    if result_key is not None:
        if not isinstance(result_key, list):
            result_key = [result_key]
        result_key = [jmespath.compile(rk) for rk in result_key]
        return result_key