import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _record_non_aggregate_key_values(self, response):
    non_aggregate_keys = {}
    for expression in self._non_aggregate_key_exprs:
        result = expression.search(response)
        set_value_from_jmespath(non_aggregate_keys, expression.expression, result)
    self._non_aggregate_part = non_aggregate_keys