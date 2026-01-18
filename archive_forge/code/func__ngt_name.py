import re
from oslo_log import log as logging
from oslo_utils import encodeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
def _ngt_name(self, name):
    if name:
        return name
    return re.sub('[^a-zA-Z0-9-]', '', self.physical_resource_name())