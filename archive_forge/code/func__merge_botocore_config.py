import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def _merge_botocore_config(config_a, config_b):
    """
    Merges the extra configuration options from config_b into config_a.
    Supports both botocore.config.Config objects and dicts
    """
    if not config_b:
        return config_a
    if not isinstance(config_b, botocore.config.Config):
        config_b = botocore.config.Config(**config_b)
    return config_a.merge(config_b)