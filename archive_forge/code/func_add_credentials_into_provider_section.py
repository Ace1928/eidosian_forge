import copy
import logging
import os
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.util import check_legacy_fields
def add_credentials_into_provider_section(config):
    provider_config = config['provider']
    if 'vsphere_config' in provider_config and 'credentials' in provider_config['vsphere_config']:
        return
    env_credentials = {'server': os.environ['VSPHERE_SERVER'], 'user': os.environ['VSPHERE_USER'], 'password': os.environ['VSPHERE_PASSWORD']}
    provider_config['vsphere_config']['credentials'] = env_credentials