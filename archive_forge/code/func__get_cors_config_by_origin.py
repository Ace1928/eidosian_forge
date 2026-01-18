import copy
import logging
import debtcollector
from oslo_config import cfg
from oslo_middleware import base
import webob.exc
def _get_cors_config_by_origin(self, origin):
    if origin not in self.allowed_origins:
        if '*' in self.allowed_origins:
            origin = '*'
        else:
            LOG.debug("CORS request from origin '%s' not permitted." % origin)
            raise InvalidOriginError(origin)
    return (origin, self.allowed_origins[origin])