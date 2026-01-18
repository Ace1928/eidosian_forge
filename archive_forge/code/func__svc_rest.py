from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
def _svc_rest(self, method, headers, cmd, cmdopts, cmdargs, timeout=10):
    """ Run SVC command with token info added into header
        :param method: http method, POST or GET
        :type method: string
        :param headers: http headers
        :type headers: dict
        :param cmd: svc command to run
        :type cmd: string
        :param cmdopts: svc command options, name paramter and value
        :type cmdopts: dict
        :param cmdargs: svc command arguments, non-named paramaters
        :type timeout: int
        :param timeout: open_url argument to set timeout for http gateway
        :return: dict of command results
        :rtype: dict
        """
    r = {'url': None, 'code': None, 'err': None, 'out': None, 'data': None}
    postfix = cmd
    if cmdargs:
        postfix = '/'.join([postfix] + [quote(str(a)) for a in cmdargs])
    url = '/'.join([self.resturl] + [postfix])
    r['url'] = url
    self.log('_svc_rest: url=%s', url)
    payload = cmdopts if cmdopts else None
    data = self.module.jsonify(payload).encode('utf8')
    r['data'] = cmdopts
    self.log('_svc_rest: payload=%s', payload)
    try:
        o = open_url(url, method=method, headers=headers, timeout=timeout, validate_certs=self.validate_certs, data=bytes(data))
    except HTTPError as e:
        self.log('_svc_rest: httperror %s', str(e))
        r['code'] = e.getcode()
        r['out'] = e.read()
        r['err'] = ('HTTPError %s', str(e))
        return r
    except Exception as e:
        self.log('_svc_rest: exception : %s', str(e))
        r['err'] = ('Exception %s', str(e))
        return r
    try:
        j = json.load(o)
    except ValueError as e:
        self.log('_svc_rest: value error pass: %s', str(e))
        return r
    r['out'] = j
    return r