from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
@classmethod
def SpecAndAnnotationsOnly(cls, execution):
    """Special wrapper for spec only that also covers metadata annotations.

      For a message type without its own metadata, like TaskTemplateSpec,
      metadata fields should either raise AttributeErrors or refer to the
      metadata of a different message depending on use case. This method handles
      the annotations of metadata by referencing the parent job's annotations.
      All other metadata fields will fall through to k8s_object which will
      lead to AttributeErrors.

      Args:
        execution: The parent job for this InstanceTemplateSpec

      Returns:
        A new k8s_object to wrap the TaskTemplateSpec with only the spec
        fields and the metadata annotations.
      """
    spec_wrapper = super(Execution.TaskTemplateSpec, cls).SpecOnly(execution.spec.template.spec, execution.MessagesModule())
    spec_wrapper._annotations = execution.annotations
    return spec_wrapper