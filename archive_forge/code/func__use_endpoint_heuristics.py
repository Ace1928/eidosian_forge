import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def _use_endpoint_heuristics():
    env_var = os.environ.get('BOTO_USE_ENDPOINT_HEURISTICS', 'false').lower()
    config_var = boto.config.getbool('Boto', 'use_endpoint_heuristics', False)
    return env_var == 'true' or config_var