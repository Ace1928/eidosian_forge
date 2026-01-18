from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
def _TransformKnownFields(self, printer, registration):
    for field in self._KNOWN_FIELDS_BY_IMPORTANCE:
        record = getattr(registration, field, None)
        if record:
            printer.AddRecord({field: record}, delimit=False)