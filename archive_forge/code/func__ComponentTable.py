from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
def _ComponentTable(self, record):
    rows = [(x.name, str(x.event_input), x.description) for x in record.components]
    return cp.Table([('NAME', 'TAKES CE-INPUT', 'DESCRIPTION')] + rows)