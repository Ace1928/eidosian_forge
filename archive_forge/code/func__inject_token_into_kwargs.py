import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _inject_token_into_kwargs(self, op_kwargs, next_token):
    for name, token in next_token.items():
        if token is not None and token != 'None':
            op_kwargs[name] = token
        elif name in op_kwargs:
            del op_kwargs[name]