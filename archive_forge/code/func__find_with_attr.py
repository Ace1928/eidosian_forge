from oslo_config import cfg
from oslo_utils import uuidutils
from glanceclient import client as gc
from glanceclient import exc
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
from heat.engine import constraints
def _find_with_attr(self, entity, **kwargs):
    """Find a item for entity with attributes matching ``**kwargs``."""
    matches = list(self._findall_with_attr(entity, **kwargs))
    num_matches = len(matches)
    if num_matches == 0:
        raise client_exception.EntityMatchNotFound(entity=entity, args=kwargs)
    elif num_matches > 1:
        raise client_exception.EntityUniqueMatchNotFound(entity=entity, args=kwargs)
    else:
        return matches[0]