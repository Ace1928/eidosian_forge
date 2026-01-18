import email
from email.mime import multipart
from email.mime import text
import os
from oslo_utils import uuidutils
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config
from heat.engine import support
from heat.rpc import api as rpc_api
@staticmethod
def _append_part(subparts, part, subtype, filename):
    if not subtype and filename:
        subtype = os.path.splitext(filename)[0]
    msg = MultipartMime._create_message(part, subtype, filename)
    subparts.append(msg)