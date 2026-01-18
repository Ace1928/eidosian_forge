from __future__ import absolute_import
import re
def MoveFieldsTo(field_names, target_field_name):
    target = {}
    for field_name in field_names:
        if field_name in automatic_scaling:
            target[field_name] = automatic_scaling[field_name]
            del automatic_scaling[field_name]
    if target:
        automatic_scaling[target_field_name] = target