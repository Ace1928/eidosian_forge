import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def _create_auth_spec(module=None, **kwargs) -> Dict:
    auth: Dict = {}
    for true_name, arg_name in AUTH_ARG_MAP.items():
        if module and module.params.get(arg_name) is not None:
            auth[true_name] = module.params.get(arg_name)
        elif arg_name in kwargs and kwargs.get(arg_name) is not None:
            auth[true_name] = kwargs.get(arg_name)
        elif true_name in kwargs and kwargs.get(true_name) is not None:
            auth[true_name] = kwargs.get(true_name)
        elif arg_name == 'proxy_headers':
            proxy_headers = {}
            for key in AUTH_PROXY_HEADERS_SPEC.keys():
                env_value = os.getenv('K8S_AUTH_PROXY_HEADERS_{0}'.format(key.upper()), None)
                if env_value is not None:
                    if AUTH_PROXY_HEADERS_SPEC[key].get('type') == 'bool':
                        env_value = env_value.lower() not in ['0', 'false', 'no']
                    proxy_headers[key] = env_value
            if proxy_headers is not {}:
                auth[true_name] = proxy_headers
        else:
            env_value = os.getenv('K8S_AUTH_{0}'.format(arg_name.upper()), None) or os.getenv('K8S_AUTH_{0}'.format(true_name.upper()), None)
            if env_value is not None:
                if AUTH_ARG_SPEC[arg_name].get('type') == 'bool':
                    env_value = env_value.lower() not in ['0', 'false', 'no']
                auth[true_name] = env_value
    return auth