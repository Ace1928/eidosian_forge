from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def _execute_request(self, path, method=None, payload=None, params=None):
    """ Execute query """
    try:
        resp, info = fetch_url(self.module, self.url, headers=self.headers, data=payload, method=self.method, timeout=self.params['timeout'], use_proxy=self.params['use_proxy'])
        self.status = info['status']
        if self.status == 429:
            self.retry += 1
            if self.retry <= 10:
                try:
                    self.module.warn('Rate limiter hit, retry {0}...pausing for {1} seconds'.format(self.retry, info['Retry-After']))
                    time.sleep(info['Retry-After'])
                except KeyError:
                    self.module.warn('Rate limiter hit, retry {0}...pausing for 5 seconds'.format(self.retry))
                    time.sleep(5)
                return self._execute_request(path, method=method, payload=payload, params=params)
            else:
                self.fail_json(msg='Rate limit retries failed for {url}'.format(url=self.url))
        elif self.status == 500:
            self.retry += 1
            self.module.warn('Internal server error 500, retry {0}'.format(self.retry))
            if self.retry <= 10:
                self.retry_time += self.retry * INTERNAL_ERROR_RETRY_MULTIPLIER
                time.sleep(self.retry_time)
                return self._execute_request(path, method=method, payload=payload, params=params)
            else:
                self.fail_json(msg='Rate limit retries failed for {url}'.format(url=self.url))
        elif self.status == 502:
            self.module.warn('Internal server error 502, retry {0}'.format(self.retry))
        elif self.status == 400:
            raise HTTPError('')
        elif self.status >= 400:
            self.fail_json(msg=self.status, url=self.url)
            raise HTTPError('')
    except HTTPError:
        try:
            self.fail_json(msg='HTTP error {0} - {1} - {2}'.format(self.status, self.url, json.loads(info['body'])['errors'][0]))
        except json.decoder.JSONDecodeError:
            self.fail_json(msg='HTTP error {0} - {1}'.format(self.status, self.url))
    self.retry = 0
    return (resp, info)