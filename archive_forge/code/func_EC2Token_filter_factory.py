import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
import requests
import webob
from heat.api.aws import exception
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
def EC2Token_filter_factory(global_conf, **local_conf):
    """Factory method for paste.deploy."""
    conf = global_conf.copy()
    conf.update(local_conf)

    def filter(app):
        return EC2Token(app, conf)
    return filter