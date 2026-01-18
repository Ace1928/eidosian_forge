import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _get_output_tokens(self, config):
    output = []
    output_token = config['output_token']
    if not isinstance(output_token, list):
        output_token = [output_token]
    for config in output_token:
        output.append(jmespath.compile(config))
    return output