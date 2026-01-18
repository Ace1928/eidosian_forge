from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformPipelineRef(pr):
    """Convert PipelineRef into Tekton yaml."""
    pipeline_ref = {}
    if 'name' in pr:
        pipeline_ref['name'] = pr.pop('name')
    if 'resolver' in pr:
        pipeline_ref['resolver'] = pr.pop('resolver')
    if 'params' in pr:
        pipeline_ref['params'] = _TransformParams(pr.pop('params'))
    return pipeline_ref