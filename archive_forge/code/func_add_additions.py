from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def add_additions(result, updated_soa, updated_record):
    if updated_soa:
        result['additions'].append(updated_soa)
    if updated_record:
        result['additions'].append(updated_record)