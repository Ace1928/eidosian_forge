from __future__ import (absolute_import, division, print_function)
import hashlib
import json
import re
import uuid
import os
from collections import namedtuple
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.six import iteritems
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
from ansible.errors import AnsibleParserError, AnsibleError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native, to_bytes, to_text
from itertools import chain
class AzureRMRestConfiguration(Configuration):

    def __init__(self, credentials, subscription_id, base_url=None):
        if credentials is None:
            raise ValueError("Parameter 'credentials' must not be None.")
        if subscription_id is None:
            raise ValueError("Parameter 'subscription_id' must not be None.")
        if not base_url:
            base_url = 'https://management.azure.com'
        credential_scopes = base_url + '/.default'
        super(AzureRMRestConfiguration, self).__init__()
        self.authentication_policy = BearerTokenCredentialPolicy(credentials, credential_scopes)
        self.credentials = credentials
        self.subscription_id = subscription_id