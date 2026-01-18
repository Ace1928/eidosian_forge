from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def WithServiceYaml(self, yaml_path):
    """Use a pre-written service yaml for deployment."""
    service_dict = yaml.load_path(yaml_path)
    if 'status' in service_dict:
        del service_dict['status']
    metadata = yaml_helper.GetOrCreate(service_dict, ['metadata'])
    namespace = metadata.get('namespace', None)
    if namespace is not None and (not isinstance(namespace, str)):
        service_dict['metadata']['namespace'] = str(namespace)
    try:
        service = messages_util.DictToMessageWithErrorCheck(service_dict, RUN_MESSAGES_MODULE.Service)
    except messages_util.ScalarTypeMismatchError as e:
        exceptions.MaybeRaiseCustomFieldMismatch(e, help_text='Please make sure that the YAML file matches the Knative service definition spec in https://kubernetes.io/docs/reference/kubernetes-api/service-resources/service-v1/#Service.')
    if self.project:
        service.metadata.namespace = str(self.project)
    replacements = {'service': service}
    container = service.spec.template.spec.containers[0]
    replacements['image'] = container.image
    if container.resources and container.resources.limits:
        for limit in container.resources.limits.additionalProperties:
            replacements[limit.key] = limit.value
    if service.metadata.name:
        replacements['service_name'] = service.metadata.name
    return self.replace(**replacements)