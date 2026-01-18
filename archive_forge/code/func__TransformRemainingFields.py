from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
def _TransformRemainingFields(self, printer, registration):
    for field in self._KNOWN_FIELDS_BY_IMPORTANCE:
        if getattr(registration, field, None):
            self._ClearField(registration, field)
    finished = True
    if registration.all_unrecognized_fields():
        finished = False
    for f in registration.all_fields():
        if getattr(registration, f.name):
            finished = False
    if not finished:
        printer.AddRecord(registration, delimit=False)