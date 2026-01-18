from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _TransformTaskRunResults(rs):
    """Convert TaskRunResults into Tekton yaml."""
    results = []
    for r in rs:
        result = {}
        if 'name' in r:
            result['name'] = r.pop('name')
        if 'resultValue' in r:
            result['value'] = _TransformResultValue(r.pop('resultValue'))
        results.append(result)
    return results