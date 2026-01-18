import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_endpoint_url_config_global(self):
    return self._scoped_config.get('endpoint_url')