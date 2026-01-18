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
def _append_multiparts(subparts, multi_part):
    multi_parts = email.message_from_string(multi_part)
    if not multi_parts or not multi_parts.is_multipart():
        return
    for part in multi_parts.get_payload():
        MultipartMime._append_part(subparts, part.get_payload(), part.get_content_subtype(), part.get_filename())