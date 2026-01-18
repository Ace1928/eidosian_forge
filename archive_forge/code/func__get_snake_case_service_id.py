import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
def _get_snake_case_service_id(self, client_name):
    client_name = utils.SERVICE_NAME_ALIASES.get(client_name, client_name)
    hyphenized_service_id = utils.CLIENT_NAME_TO_HYPHENIZED_SERVICE_ID_OVERRIDES.get(client_name, client_name)
    return hyphenized_service_id.replace('-', '_')