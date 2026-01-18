from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PipelineTemplateMetadata(_messages.Message):
    """Pipeline template metadata if PipelineJob.template_uri is from supported
  template registry. Currently, the only supported registry is Artifact
  Registry.

  Fields:
    version: The version_name in artifact registry. Will always be presented
      in output if the PipelineJob.template_uri is from supported template
      registry. Format is "sha256:abcdef123456...".
  """
    version = _messages.StringField(1)