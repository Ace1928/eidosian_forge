import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
class PaginatorModel:

    def __init__(self, paginator_config):
        self._paginator_config = paginator_config['pagination']

    def get_paginator(self, operation_name):
        try:
            single_paginator_config = self._paginator_config[operation_name]
        except KeyError:
            raise ValueError('Paginator for operation does not exist: %s' % operation_name)
        return single_paginator_config