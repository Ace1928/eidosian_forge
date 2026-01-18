import copy
from botocore.utils import merge_dicts
def _merge_client_retry_config(retry_config, client_retry_config):
    max_retry_attempts_override = client_retry_config.get('max_attempts')
    if max_retry_attempts_override is not None:
        retry_config['__default__']['max_attempts'] = max_retry_attempts_override + 1