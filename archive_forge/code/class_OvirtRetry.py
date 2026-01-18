from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
class OvirtRetry(CloudRetry):
    base_class = _sdk4_error_maybe()

    @staticmethod
    def status_code_from_exception(error):
        return error.code

    @staticmethod
    def found(response_code, catch_extra_error_codes=None):
        retry_on = [409]
        if catch_extra_error_codes:
            retry_on.extend(catch_extra_error_codes)
        return response_code in retry_on