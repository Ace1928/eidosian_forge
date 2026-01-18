import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _set_s3UsEast1RegionalEndpoints(self, config_store, value):
    self._update_section_provider(config_store, 's3', 'us_east_1_regional_endpoint', value)