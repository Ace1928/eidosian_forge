from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import custom_printer_base
def _WorkflowDisplayLines(self, workflow):
    """Apply formatting to the workflow for describe command."""
    if 'pipelineSpecYaml' in workflow:
        yaml_str = workflow.pop('pipelineSpecYaml')
    elif 'pipelineSpec' in workflow and 'generatedYaml' in workflow['pipelineSpec']:
        yaml_str = workflow['pipelineSpec'].pop('generatedYaml')
        del workflow['pipelineSpec']
    else:
        return
    data = yaml.load(yaml_str, round_trip=True)
    workflow['pipeline'] = {'spec': data}
    yaml_str = yaml.dump(workflow, round_trip=True)
    return custom_printer_base.Lines(yaml_str.split('\n'))