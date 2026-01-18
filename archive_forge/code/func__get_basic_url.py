from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _get_basic_url(self, is_perobject):
    url_libs = ['']
    if is_perobject:
        url_libs = self.perobject_jrpc_urls
    else:
        url_libs = self.jrpc_urls
    the_url = None
    if 'adom' in self.url_params and (not url_libs[0].endswith('{adom}')):
        adom = self.module.params['adom']
        if adom == 'global':
            for url in url_libs:
                if '/global/' in url and '/adom/{adom}/' not in url:
                    the_url = url
                    break
            if not the_url:
                self.module.fail_json(msg='No global url for the request, please use other adom.')
        else:
            for url in url_libs:
                if '/adom/{adom}/' in url:
                    the_url = url
                    break
            if not the_url:
                self.module.fail_json(msg='No url for the requested adom:%s, please use other adom.' % adom)
    else:
        the_url = url_libs[0]
    if not the_url:
        raise AssertionError('the_url is not expected to be NULL')
    return self._get_replaced_url(the_url)