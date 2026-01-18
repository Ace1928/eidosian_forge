import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def _boto3_conn(conn_type=None, resource=None, region=None, endpoint=None, **params):
    """
    Builds a boto3 resource/client connection cleanly wrapping the most common failures.
    No exceptions are caught/handled.
    """
    profile = params.pop('profile_name', None)
    if conn_type not in ['both', 'resource', 'client']:
        raise ValueError('There is an issue in the calling code. You must specify either both, resource, or client to the conn_type parameter in the boto3_conn function call')
    config = botocore.config.Config(user_agent=_get_user_agent_string())
    for param in ('config', 'aws_config'):
        config = _merge_botocore_config(config, params.pop(param, None))
    session = boto3.session.Session(profile_name=profile)
    enable_placebo(session)
    if conn_type == 'resource':
        return session.resource(resource, config=config, region_name=region, endpoint_url=endpoint, **params)
    elif conn_type == 'client':
        return session.client(resource, config=config, region_name=region, endpoint_url=endpoint, **params)
    else:
        client = session.client(resource, config=config, region_name=region, endpoint_url=endpoint, **params)
        resource = session.resource(resource, config=config, region_name=region, endpoint_url=endpoint, **params)
        return (client, resource)