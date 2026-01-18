from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformResultValue(v):
    """Convert ResultValue into Tekton yaml."""
    if 'stringVal' in v:
        return v.pop('stringVal')
    if 'arrayVal' in v:
        return v.pop('arrayVal')
    if 'objectVal' in v:
        return v.pop('objectVal')
    return v