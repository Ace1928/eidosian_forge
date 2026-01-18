from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
def _get_from_secret(self, key):
    result = super(RemoteStack, self).client_plugin('barbican').get_secret_payload_by_ref(secret_ref='secrets/%s' % key)
    return result