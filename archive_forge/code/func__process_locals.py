from collections import ChainMap
from jinja2.utils import missing
from ansible.errors import AnsibleError, AnsibleUndefinedVariable
from ansible.module_utils.common.text.converters import to_native
def _process_locals(_l):
    if _l is None:
        return {}
    return {k: v for k, v in _l.items() if v is not missing and k not in {'context', 'environment', 'template'}}