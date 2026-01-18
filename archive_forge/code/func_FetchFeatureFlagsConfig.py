from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import hashlib
import logging
import os
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
def FetchFeatureFlagsConfig():
    """Downloads the feature flag config file."""
    import requests
    from googlecloudsdk.core import requests as core_requests
    try:
        yaml_request = core_requests.GetSession()
        response = yaml_request.get(_FEATURE_FLAG_YAML_URL)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.debug('Unable to fetch feature flags config from [%s]: %s', _FEATURE_FLAG_YAML_URL, e)
    return None