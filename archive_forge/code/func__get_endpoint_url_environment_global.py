import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_endpoint_url_environment_global(self):
    return EnvironmentProvider(name='AWS_ENDPOINT_URL', env=self._environ).provide()