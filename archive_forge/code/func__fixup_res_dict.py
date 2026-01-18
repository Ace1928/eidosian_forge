import collections
import contextlib
import hashlib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from webob import exc as web_exc
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib.api.definitions import network as net_apidef
from neutron_lib.api.definitions import port as port_apidef
from neutron_lib.api.definitions import portbindings as pb
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib.api.definitions import subnet as subnet_apidef
from neutron_lib import constants
from neutron_lib import exceptions
def _fixup_res_dict(context, attr_name, res_dict, check_allow_post=True):
    attr_info = attributes.RESOURCES[attr_name]
    attr_ops = attributes.AttributeInfo(attr_info)
    try:
        attr_ops.populate_project_id(context, res_dict, True)
        attributes.populate_project_info(attr_info)
        attr_ops.verify_attributes(res_dict)
    except web_exc.HTTPBadRequest as e:
        raise ValueError(e.detail) from e
    attr_ops.fill_post_defaults(res_dict, check_allow_post=check_allow_post)
    attr_ops.convert_values(res_dict)
    return res_dict