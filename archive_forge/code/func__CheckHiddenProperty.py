from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.diagnostics import check_base
from googlecloudsdk.core.diagnostics import diagnostic_base
import six
def _CheckHiddenProperty(self, prop):
    if six.text_type(prop) in self._ALLOWLIST:
        return
    if not self.ignore_hidden_property_allowlist and six.text_type(prop) in self.allowlist:
        return
    value = properties._GetPropertyWithoutCallback(prop, self._properties_file)
    if value is not None:
        msg = '[{0}]'.format(prop)
        return check_base.Failure(message=msg)