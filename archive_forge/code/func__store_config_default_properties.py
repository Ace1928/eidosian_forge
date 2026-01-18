from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def _store_config_default_properties(self, attributes=None):
    """Method for storing default values of properties in resource data.

        Some properties have default values, specified in project configuration
        file, so cannot be hardcoded into properties_schema, but should be
        stored for further using. So need to get created resource and take
        required property's value.
        """
    if attributes is None:
        attributes = self._show_resource()
    if attributes.get('volume_type') is not None:
        self.data_set(self.VOLUME_TYPE, attributes['volume_type'])
    else:
        self.data_delete(self.VOLUME_TYPE)