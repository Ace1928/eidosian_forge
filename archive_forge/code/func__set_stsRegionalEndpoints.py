import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _set_stsRegionalEndpoints(self, config_store, value):
    self._update_provider(config_store, 'sts_regional_endpoints', value)