from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _operation_results(self, response, data, handle=None):
    """
        Builds the results for an operation from task, job, or action response.

        :param response: HTTP response object
        :param data: HTTP response data
        :param handle: The task or job handle that was last used
        :return: dict containing operation results
        """
    operation_results = {'status': None, 'messages': [], 'handle': None, 'ret': True, 'resets_requested': []}
    if response.status == 204:
        operation_results['status'] = 'Completed'
    else:
        operation_results['handle'] = handle
        if response.status == 202:
            operation_results['handle'] = response.getheader('Location', handle)
        if data is not None:
            response_type = data.get('@odata.type', '')
            if response_type.startswith('#Task.') or response_type.startswith('#Job.'):
                operation_results['status'] = data.get('TaskState', data.get('JobState'))
                operation_results['messages'] = data.get('Messages', [])
            else:
                operation_results['status'] = 'Completed'
                if response.status >= 400:
                    operation_results['status'] = 'Exception'
                operation_results['messages'] = data.get('error', {}).get('@Message.ExtendedInfo', [])
        else:
            operation_results['status'] = 'Completed'
            if response.status == 202:
                operation_results['status'] = 'New'
            elif response.status >= 400:
                operation_results['status'] = 'Exception'
        if operation_results['status'] in ['Completed', 'Cancelled', 'Exception', 'Killed']:
            operation_results['handle'] = None
        for message in operation_results['messages']:
            message_id = message.get('MessageId')
            if message_id is None:
                continue
            if message_id.startswith('Update.1.') and message_id.endswith('.OperationTransitionedToJob'):
                operation_results['status'] = 'New'
                operation_results['handle'] = message['MessageArgs'][0]
                operation_results['resets_requested'] = []
                break
            if message_id.startswith('Base.1.') and message_id.endswith('.ResetRequired'):
                reset = {'uri': message['MessageArgs'][0], 'type': message['MessageArgs'][1]}
                operation_results['resets_requested'].append(reset)
    return operation_results