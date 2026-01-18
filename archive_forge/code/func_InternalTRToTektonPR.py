from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def InternalTRToTektonPR(self, internal):
    """Convert Internal TR into Tekton yaml."""
    tr = {'metadata': {}, 'spec': {}, 'status': {}}
    if 'name' in internal:
        tr['metadata']['name'] = output_util.ParseName(internal.pop('name'), 'taskrun')
    if 'params' in internal:
        tr['spec']['params'] = _TransformParams(internal.pop('params'))
    if 'taskSpec' in internal:
        tr['spec']['taskSpec'] = _TransformTaskSpec(internal.pop('taskSpec'))
    elif 'taskRef' in internal:
        tr['spec']['taskRef'] = _TransformTaskRef(internal.pop('taskRef'))
    if 'timeout' in internal:
        tr['spec']['timeout'] = internal.pop('timeout')
    if 'workspaces' in internal:
        tr['spec']['workspaces'] = internal.pop('workspaces')
    if 'serviceAccountName' in internal:
        tr['spec']['serviceAccountName'] = internal.pop('serviceAccountName')
    if 'conditions' in internal:
        tr['status']['conditions'] = _TransformConditions(internal.pop('conditions'))
    if 'startTime' in internal:
        tr['status']['startTime'] = internal.pop('startTime')
    if 'completionTime' in internal:
        tr['status']['completionTime'] = internal.pop('completionTime')
    if 'resolvedTaskSpec' in internal:
        rts = internal.pop('resolvedTaskSpec')
        tr['status']['taskSpec'] = _TransformTaskSpec(rts)
    if 'steps' in internal:
        tr['status']['steps'] = internal.pop('steps')
    if 'results' in internal:
        tr['status']['results'] = _TransformTaskRunResults(internal.pop('results'))
    if 'sidecars' in internal:
        tr['status']['sidecars'] = internal.pop('sidecars')
    return tr