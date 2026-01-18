import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_endpoint_url_config_service(self):
    snakecase_service_id = self._transformed_service_id.lower()
    return self._get_services_config().get(snakecase_service_id, {}).get('endpoint_url')