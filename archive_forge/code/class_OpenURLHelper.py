from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url, open_url, NoSSLError, ConnectionError
import ansible.module_utils.six.moves.urllib.error as urllib_error
class OpenURLHelper(HTTPHelper):

    def fetch_url(self, url, method='GET', headers=None, data=None, timeout=None):
        info = {}
        try:
            req = open_url(url, method=method, headers=headers, data=data, timeout=timeout)
            result = req.read()
            info.update(dict(((k.lower(), v) for k, v in req.info().items())))
            info['status'] = req.code
            info['url'] = req.geturl()
            req.close()
        except urllib_error.HTTPError as e:
            try:
                result = e.read()
            except AttributeError:
                result = ''
            try:
                info.update(dict(((k.lower(), v) for k, v in e.info().items())))
            except Exception:
                pass
            info['status'] = e.code
        except NoSSLError as e:
            raise NetworkError('Cannot connect via SSL: {0}'.format(to_native(e)))
        except (ConnectionError, ValueError) as e:
            raise NetworkError('Connection error: {0}'.format(to_native(e)))
        return (result, info)