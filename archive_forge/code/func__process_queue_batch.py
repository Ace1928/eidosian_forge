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
def _process_queue_batch(self):
    while True:
        batch_requests = []
        batch_item_index = 0
        batch_response_handlers = dict()
        try:
            while batch_item_index < 100:
                item = self._request_queue.get_nowait()
                name = str(uuid.uuid4())
                query_parameters = {'api-version': item.api_version}
                header_parameters = {'x-ms-client-request-id': str(uuid.uuid4()), 'Content-Type': 'application/json; charset=utf-8'}
                body = {}
                req = self.new_client.get(item.url, query_parameters, header_parameters, body)
                batch_requests.append(dict(httpMethod='GET', url=req.url, name=name))
                batch_response_handlers[name] = item
                batch_item_index += 1
        except Empty:
            pass
        if not batch_requests:
            break
        batch_resp = self._send_batch(batch_requests)
        key_name = None
        if 'responses' in batch_resp:
            key_name = 'responses'
        elif 'value' in batch_resp:
            key_name = 'value'
        else:
            raise AnsibleError("didn't find expected key responses/value in batch response")
        for idx, r in enumerate(batch_resp[key_name]):
            status_code = r.get('httpStatusCode')
            returned_name = r['name']
            result = batch_response_handlers[returned_name]
            if status_code == 200:
                result.handler(r['content'], **result.handler_args)