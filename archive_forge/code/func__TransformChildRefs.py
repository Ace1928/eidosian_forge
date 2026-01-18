from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformChildRefs(crs):
    """Convert ChildReferences into Tekton yaml."""
    child_refs = []
    for cr in crs:
        child_ref = {}
        if 'name' in cr:
            child_ref['name'] = cr.pop('name')
        if 'pipelineTask' in cr:
            child_ref['pipelineTask'] = cr.pop('pipelineTask')
        child_refs.append(child_ref)
    return child_refs