import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def _create_configuration(auth: Dict):

    def auth_set(*names: list) -> bool:
        return all((auth.get(name) for name in names))
    if auth_set('host'):
        auth['host'] = auth.get('host').rstrip('/')
    if auth_set('username', 'password', 'host') or auth_set('api_key', 'host') or auth_set('cert_file', 'key_file', 'host'):
        pass
    elif auth_set('kubeconfig') or auth_set('context'):
        try:
            _load_config(auth)
        except Exception as err:
            raise err
    else:
        try:
            kubernetes.config.load_incluster_config()
        except kubernetes.config.ConfigException:
            try:
                _load_config(auth)
            except Exception as err:
                raise err
    try:
        configuration = kubernetes.client.Configuration().get_default_copy()
    except AttributeError:
        configuration = kubernetes.client.Configuration()
    for key, value in iteritems(auth):
        if key in AUTH_ARG_MAP.keys() and value is not None:
            if key == 'api_key':
                setattr(configuration, key, {'authorization': 'Bearer {0}'.format(value)})
            elif key == 'proxy_headers':
                headers = urllib3.util.make_headers(**value)
                setattr(configuration, key, headers)
            else:
                setattr(configuration, key, value)
    return configuration