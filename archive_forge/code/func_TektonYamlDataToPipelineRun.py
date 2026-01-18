from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import log
def TektonYamlDataToPipelineRun(data):
    """Convert Tekton yaml file into PipelineRun message."""
    _VersionCheck(data)
    _MetadataTransform(data)
    spec = data['spec']
    if 'pipelineSpec' in spec:
        _PipelineSpecTransform(spec['pipelineSpec'])
    elif 'pipelineRef' in spec:
        input_util.RefTransform(spec['pipelineRef'])
    else:
        raise cloudbuild_exceptions.InvalidYamlError('PipelineSpec or PipelineRef is required.')
    if 'resources' in spec:
        spec.pop('resources')
        log.warning('PipelineResources are dropped because they are deprecated: https://github.com/tektoncd/pipeline/blob/main/docs/resources.md')
    _ServiceAccountTransformPipelineSpec(spec)
    input_util.ParamDictTransform(spec.get('params', []))
    messages = client_util.GetMessagesModule()
    schema_message = encoding.DictToMessage(spec, messages.PipelineRun)
    input_util.UnrecognizedFields(schema_message)
    return schema_message