from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformConditions(cs):
    """Convert Conditions into Tekton yaml."""
    conditions = []
    for c in cs:
        condition = {}
        if 'message' in c:
            condition['message'] = c.pop('message')
            if 'lastTransitionTime' in c:
                condition['lastTransitionTime'] = c.pop('lastTransitionTime')
            if 'status' in c:
                condition['status'] = c.pop('status').capitalize()
            if 'type' in c:
                condition['type'] = c.pop('type').capitalize()
            if 'reason' in c:
                condition['reason'] = c.pop('reason')
                conditions.append(condition)
    return conditions