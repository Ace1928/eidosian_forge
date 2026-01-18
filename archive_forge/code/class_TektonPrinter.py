from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild.v2 import output_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
class TektonPrinter(custom_printer_base.CustomPrinterBase):
    """Print a  PipelineRun or TaskRun in Tekton YAML format."""

    def Transform(self, internal_proto):
        proto = encoding.MessageToDict(internal_proto)
        if 'pipelineSpec' in proto or 'pipelineRef' in proto:
            yaml_str = self.InternalPRToTektonPR(proto)
            return yaml.dump(yaml_str, round_trip=True)
        elif 'taskSpec' in proto or 'taskRef' in proto:
            yaml_str = self.InternalTRToTektonPR(proto)
            return yaml.dump(yaml_str, round_trip=True)

    def InternalPRToTektonPR(self, internal):
        """Convert Tekton yaml file into PipelineRun message."""
        pr = {'metadata': {}, 'spec': {}, 'status': {}}
        if 'name' in internal:
            pr['metadata']['name'] = output_util.ParseName(internal.pop('name'), 'pipelinerun')
        if 'annotations' in internal:
            pr['metadata']['annotations'] = internal.pop('annotations')
        if 'params' in internal:
            pr['spec']['params'] = _TransformParams(internal.pop('params'))
        if 'pipelineSpec' in internal:
            pr['spec']['pipelineSpec'] = _TransformPipelineSpec(internal.pop('pipelineSpec'))
        elif 'pipelineRef' in internal:
            pr['spec']['pipelineRef'] = _TransformPipelineRef(internal.pop('pipelineRef'))
        if 'timeout' in internal:
            pr['spec']['timeout'] = internal.pop('timeout')
        if 'workspaces' in internal:
            pr['spec']['workspaces'] = internal.pop('workspaces')
        if 'conditions' in internal:
            conditions = internal.pop('conditions')
            pr['status']['conditions'] = _TransformConditions(conditions)
        if 'startTime' in internal:
            pr['status']['startTime'] = internal.pop('startTime')
        if 'completionTime' in internal:
            pr['status']['completionTime'] = internal.pop('completionTime')
        if 'resolvedPipelineSpec' in internal:
            rps = internal.pop('resolvedPipelineSpec')
            pr['status']['pipelineSpec'] = _TransformPipelineSpec(rps)
        if 'results' in internal:
            pr['status']['results'] = _TransformPipelineRunResults(internal.pop('results'))
        if 'childReferences' in internal:
            crs = internal.pop('childReferences')
            pr['status']['childReferences'] = crs
        if 'serviceAccount' in internal:
            pr['taskRunTemplate'] = {'serviceAccountName': internal.pop('serviceAccount')}
        return pr

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