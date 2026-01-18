from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
def ReadySymbolAndColor(self):
    """Return a tuple of ready_symbol and display color for this object."""
    encoding = console_attr.GetConsoleAttr().GetEncoding()
    if self.running_state == 'Running':
        return (self._PickSymbol('…', '.', encoding), 'yellow')
    elif self.running_state == 'Succeeded':
        return (self._PickSymbol('✔', '+', encoding), 'green')
    elif self.running_state == 'Failed':
        return ('X', 'red')
    elif self.running_state == 'Cancelled':
        return ('!', 'yellow')
    elif self.running_state == 'Abandoned':
        return ('-', 'yellow')
    return ('.', 'yellow')