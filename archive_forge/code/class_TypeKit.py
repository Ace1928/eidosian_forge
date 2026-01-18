from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
class TypeKit(object):
    """An abstract class that represents a typekit."""

    def __init__(self, type_metadata: types_utils.TypeMetadata):
        self._type_metadata = type_metadata

    @property
    def integration_type(self):
        return self._type_metadata.integration_type

    @property
    def resource_type(self):
        return self._type_metadata.resource_type

    @property
    def is_singleton(self):
        return self._type_metadata.singleton_name is not None

    @property
    def singleton_name(self):
        return self._type_metadata.singleton_name

    @property
    def is_backing_service(self):
        return self._type_metadata.service_type == types_utils.ServiceType.BACKING

    @property
    def is_ingress_service(self):
        return self._type_metadata.service_type == types_utils.ServiceType.INGRESS

    def GetDeployMessage(self, create: bool=False) -> str:
        """Message that is shown to the user upon starting the deployment.

    Each TypeKit should override this method to at least tell the user how
    long the deployment is expected to take.

    Args:
      create: denotes if the command was a create deployment.

    Returns:
      The message displayed to the user.
    """
        del create
        if self._type_metadata.eta_in_min:
            return 'This might take up to {} minutes.'.format(self._type_metadata.eta_in_min)
        return ''

    def UpdateResourceConfig(self, parameters: Dict[str, str], resource: runapps_v1alpha1_messages.Resource) -> List[str]:
        """Updates config according to the parameters.

    Each TypeKit should override this method to update the resource config
    specific to the need of the typekit.

    Args:
      parameters: parameters from the command
      resource: the resource object of the integration

    Returns:
      list of service names referred in parameters.
    """
        metadata = self._type_metadata
        config_dict = {}
        if resource.config:
            config_dict = encoding.MessageToDict(resource.config)
        for param in metadata.parameters:
            param_value = parameters.get(param.name)
            if param_value:
                if param.data_type == 'int':
                    config_dict[param.config_name] = int(param_value)
                elif param.data_type == 'number':
                    config_dict[param.config_name] = float(param_value)
                else:
                    config_dict[param.config_name] = param_value
        resource.config = encoding.DictToMessage(config_dict, runapps_v1alpha1_messages.Resource.ConfigValue)
        return []

    def _SetBinding(self, to_resource: runapps_v1alpha1_messages.Resource, from_resource: runapps_v1alpha1_messages.Resource, parameters: Optional[Dict[str, str]]=None):
        """Add a binding from a resource to another resource.

    Args:
      to_resource: the resource this binding to be pointing to.
      from_resource: the resource this binding to be configured from.
      parameters: the binding config from parameter
    """
        from_ids = [x.targetRef.id for x in from_resource.bindings]
        if to_resource.id not in from_ids:
            from_resource.bindings.append(runapps_v1alpha1_messages.Binding(targetRef=runapps_v1alpha1_messages.ResourceRef(id=to_resource.id)))
        if parameters:
            for binding in from_resource.bindings:
                if binding.targetRef.id == to_resource.id:
                    binding_config = encoding.MessageToDict(binding.config) if binding.config else {}
                    for key in parameters:
                        binding_config[key] = parameters[key]
                    binding.config = encoding.DictToMessage(binding_config, runapps_v1alpha1_messages.Binding.ConfigValue)

    def BindServiceToIntegration(self, integration: runapps_v1alpha1_messages.Resource, workload: runapps_v1alpha1_messages.Resource, parameters: Optional[Dict[str, str]]=None):
        """Bind a workload to an integration.

    Args:
      integration: the resource of the inetgration.
      workload: the resource the workload.
      parameters: the binding config from parameter.
    """
        if self._type_metadata.service_type == types_utils.ServiceType.INGRESS:
            self._SetBinding(workload, integration, parameters)
        else:
            self._SetBinding(integration, workload, parameters)

    def UnbindServiceFromIntegration(self, integration: runapps_v1alpha1_messages.Resource, workload: runapps_v1alpha1_messages.Resource):
        """Unbind a workload from an integration.

    Args:
      integration: the resource of the inetgration.
      workload: the resource the workload.
    """
        if self._type_metadata.service_type == types_utils.ServiceType.INGRESS:
            RemoveBinding(workload, integration)
        else:
            RemoveBinding(integration, workload)

    def NewIntegrationName(self, appconfig: runapps_v1alpha1_messages.Config) -> str:
        """Returns a name for a new integration.

    Args:
      appconfig: the application config

    Returns:
      str, a new name for the integration.
    """
        name = self._GenerateIntegrationNameCandidate(self.integration_type)
        existing_names = {res.id.name for res in appconfig.resourceList if res.id.type == self.resource_type}
        while name in existing_names:
            name = self._GenerateIntegrationNameCandidate(self.integration_type)
        return name

    def _GenerateIntegrationNameCandidate(self, integration_type: str) -> str:
        """Generates a suffix for a new integration.

    Args:
      integration_type: str, name of integration.

    Returns:
      str, a new name for the integration.
    """
        integration_suffix = str(uuid.uuid4())[:4]
        name = '{}-{}'.format(integration_type, integration_suffix)
        return name

    def GetCreateSelectors(self, integration_name) -> List[Selector]:
        """Returns create selectors for given integration and service.

    Args:
      integration_name: str, name of integration.

    Returns:
      list of dict typed names.
    """
        return [{'type': self.resource_type, 'name': integration_name}]

    def GetDeleteSelectors(self, integration_name) -> List[Selector]:
        """Returns selectors for deleting the integration.

    Args:
      integration_name: str, name of integration.

    Returns:
      list of dict typed names.
    """
        return [{'type': self.resource_type, 'name': integration_name}]

    def GetBindedWorkloads(self, resource: runapps_v1alpha1_messages.Resource, all_resources: List[runapps_v1alpha1_messages.Resource], workload_type: str='service') -> List[runapps_v1alpha1_messages.ResourceID]:
        """Returns list of workloads that are associated to this resource.

    If the resource is a backing service, then it returns a list of workloads
    binding to the resource. If the resource is an ingress service, then all
    of the workloads it is binding to.

    Args:
      resource: the resource object of the integration.
      all_resources: all the resources in the application.
      workload_type: type of the workload to search for.

    Returns:
      list ResourceID of the binded workloads.
    """
        if self.is_backing_service:
            filtered_workloads = [res for res in all_resources if res.id.type == workload_type]
            return [workload.id.name for workload in filtered_workloads if FindBindings(workload, resource.id.type, resource.id.name)]
        return [res_id.targetRef.id.name for res_id in FindBindingsRecursive(resource, workload_type)]

    def GetCreateComponentTypes(self, selectors: Iterable[Selector]):
        """Returns a list of component types included in a create/update deployment.

    Args:
      selectors: list of dict of type names (string) that will be deployed.

    Returns:
      set of component types as strings. The component types can also include
      hidden resource types that should be called out as part of the deployment
      progress output.
    """
        return GetComponentTypesFromSelectors(selectors)

    def GetDeleteComponentTypes(self, selectors: Iterable[Selector]):
        """Returns a list of component types included in a delete deployment.

    Args:
      selectors: list of dict of type names (string) that will be deployed.

    Returns:
      set of component types as strings. The component types can also include
      hidden resource types that should be called out as part of the deployment
      progress output.
    """
        return GetComponentTypesFromSelectors(selectors)