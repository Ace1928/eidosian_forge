import os
from vitrageclient.common import yaml_utils
from vitrageclient import exceptions as exc
@classmethod
def _load_yaml(cls, yaml_content):
    try:
        return yaml_utils.load(yaml_content)
    except ValueError as e:
        message = 'Could not load template: %s. Reason: %s' % (yaml_content, e)
        raise exc.CommandError(message)