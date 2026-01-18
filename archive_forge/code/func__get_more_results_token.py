import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _get_more_results_token(self, config):
    more_results = config.get('more_results')
    if more_results is not None:
        return jmespath.compile(more_results)