from __future__ import (absolute_import, division, print_function)
import re
import time
import json
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
def _get_sentinel_connection(self, uri, kw):
    """
        get sentinel connection details from _uri
        """
    try:
        from redis.sentinel import Sentinel
    except ImportError:
        raise AnsibleError("The 'redis' python module (version 2.9.0 or newer) is required to use redis sentinel.")
    if ';' not in uri:
        raise AnsibleError('_uri does not have sentinel syntax.')
    connections = uri.split(';')
    connection_args = connections.pop(-1)
    if len(connection_args) > 0:
        connection_args = connection_args.split(':')
        kw['db'] = connection_args.pop(0)
        try:
            kw['password'] = connection_args.pop(0)
        except IndexError:
            pass
    sentinels = [self._parse_connection(self.re_sent_conn, shost) for shost in connections]
    display.vv('\nUsing redis sentinels: %s' % sentinels)
    scon = Sentinel(sentinels, **kw)
    try:
        return scon.master_for(self._sentinel_service_name, socket_timeout=0.2)
    except Exception as exc:
        raise AnsibleError('Could not connect to redis sentinel: %s' % to_native(exc))