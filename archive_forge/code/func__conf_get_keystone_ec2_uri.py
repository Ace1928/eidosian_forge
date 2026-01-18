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
@staticmethod
def _conf_get_keystone_ec2_uri(auth_uri):
    if auth_uri.endswith('ec2tokens'):
        return auth_uri
    if auth_uri.endswith('/'):
        return '%sec2tokens' % auth_uri
    return '%s/ec2tokens' % auth_uri