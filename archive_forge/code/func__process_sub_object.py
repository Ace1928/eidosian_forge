from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def _process_sub_object(self, all_urls, toplevel_url_token, traced_url_tokens, path, name):
    vdom = self._module.params['vdom'] if 'vdom' in self._module.params and self._module.params['vdom'] else None
    url_prefix = self.cmdb_url(path, name)
    url_suffix = ''
    if vdom == 'global':
        url_suffix = '?global=1'
    elif vdom:
        url_suffix = '?vdom=' + vdom
    for url_tokens in traced_url_tokens:
        url = dict()
        url_get = toplevel_url_token
        url_put = toplevel_url_token
        url_post = toplevel_url_token
        url_put_payload = dict()
        url_post_payload = dict()
        for token in url_tokens:
            token_name = str(list(token.keys())[0])
            token_value = str(token[token_name][0])
            token_payload = underscore_to_hyphen(token[token_name][1])
            token_islast = token == url_tokens[-1]
            if token[token_name][0]:
                url_get += '/%s/%s' % (token_name.replace('_', '-'), urlencoding.quote(token_value, safe=''))
                url_put += '/%s/%s' % (token_name.replace('_', '-'), urlencoding.quote(token_value, safe=''))
            else:
                url_get += '/%s' % token_name.replace('_', '-')
                url_put += '/%s' % token_name.replace('_', '-')
            if not token_islast:
                if token[token_name][0]:
                    url_post += '/%s/%s' % (token_name.replace('_', '-'), urlencoding.quote(token_value, safe=''))
                else:
                    url_post += '/%s' % token_name.replace('_', '-')
            else:
                url_post += '/%s' % token_name.replace('_', '-')
                url_post_payload = token_payload
                url_put_payload = token_payload
        url['get'] = url_prefix + url_get + url_suffix
        url['put'] = url_prefix + url_put + url_suffix
        url['post'] = url_prefix + url_post + url_suffix
        url['put_payload'] = url_put_payload
        url['post_payload'] = url_post_payload
        url['delete'] = url['get']
        url['vdom'] = vdom
        all_urls.append(url)