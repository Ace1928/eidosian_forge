from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
def BuildVersion(self, name, path=None, deployment_uri=None, runtime_version=None, labels=None, machine_type=None, description=None, framework=None, python_version=None, prediction_class=None, package_uris=None, accelerator_config=None, service_account=None, explanation_method=None, num_integral_steps=None, num_paths=None, image=None, command=None, container_args=None, env_vars=None, ports=None, predict_route=None, health_route=None, min_nodes=None, max_nodes=None, metrics=None, containers_hidden=True, autoscaling_hidden=True):
    """Create a Version object.

    The object is based on an optional YAML configuration file and the
    parameters to this method; any provided method parameters override any
    provided in-file configuration.

    The file may only have the fields given in
    VersionsClientBase._ALLOWED_YAML_FIELDS specified; the only parameters
    allowed are those that can be specified on the command line.

    Args:
      name: str, the name of the version object to create.
      path: str, the path to the YAML file.
      deployment_uri: str, the deploymentUri to set for the Version
      runtime_version: str, the runtimeVersion to set for the Version
      labels: Version.LabelsValue, the labels to set for the version
      machine_type: str, the machine type to serve the model version on.
      description: str, the version description.
      framework: FrameworkValueValuesEnum, the ML framework used to train this
        version of the model.
      python_version: str, The version of Python used to train the model.
      prediction_class: str, the FQN of a Python class implementing the Model
        interface for custom prediction.
      package_uris: list of str, Cloud Storage URIs containing user-supplied
        Python code to use.
      accelerator_config: an accelerator config message object.
      service_account: Specifies the service account for resource access
        control.
      explanation_method: Enables explanations and selects the explanation
        method. Valid options are 'integrated-gradients' and 'sampled-shapley'.
      num_integral_steps: Number of integral steps for Integrated Gradients and
        XRAI.
      num_paths: Number of paths for Sampled Shapley.
      image: The container image to deploy.
      command: Entrypoint for the container image.
      container_args: The command-line args to pass the container.
      env_vars: The environment variables to set on the container.
      ports: The ports to which traffic will be sent in the container.
      predict_route: The HTTP path within the container that predict requests
        are sent to.
      health_route: The HTTP path within the container that health checks are
        sent to.
      min_nodes: The minimum number of nodes to scale this model under load.
      max_nodes: The maximum number of nodes to scale this model under load.
      metrics: List of key-value pairs to set as metrics' target for
        autoscaling.
      containers_hidden: Whether or not container-related fields are hidden on
        this track.
      autoscaling_hidden: Whether or not autoscaling fields are hidden on this
        track.

    Returns:
      A Version object (for the corresponding API version).

    Raises:
      InvalidVersionConfigFile: If the file contains unexpected fields.
    """
    if path:
        allowed_fields = self._ALLOWED_YAML_FIELDS
        if not containers_hidden:
            allowed_fields |= self._CONTAINER_FIELDS
        version = self.ReadConfig(path, allowed_fields)
    else:
        version = self.version_class()
    additional_fields = {'name': name, 'deploymentUri': deployment_uri, 'runtimeVersion': runtime_version, 'labels': labels, 'machineType': machine_type, 'description': description, 'framework': framework, 'pythonVersion': python_version, 'predictionClass': prediction_class, 'packageUris': package_uris, 'acceleratorConfig': accelerator_config, 'serviceAccount': service_account}
    explanation_config = None
    if explanation_method == 'integrated-gradients':
        explanation_config = self.messages.GoogleCloudMlV1ExplanationConfig()
        ig_config = self.messages.GoogleCloudMlV1IntegratedGradientsAttribution()
        ig_config.numIntegralSteps = num_integral_steps
        explanation_config.integratedGradientsAttribution = ig_config
    elif explanation_method == 'sampled-shapley':
        explanation_config = self.messages.GoogleCloudMlV1ExplanationConfig()
        shap_config = self.messages.GoogleCloudMlV1SampledShapleyAttribution()
        shap_config.numPaths = num_paths
        explanation_config.sampledShapleyAttribution = shap_config
    elif explanation_method == 'xrai':
        explanation_config = self.messages.GoogleCloudMlV1ExplanationConfig()
        xrai_config = self.messages.GoogleCloudMlV1XraiAttribution()
        xrai_config.numIntegralSteps = num_integral_steps
        explanation_config.xraiAttribution = xrai_config
    if explanation_config is not None:
        additional_fields['explanationConfig'] = explanation_config
    if not containers_hidden:
        self._ConfigureContainer(version, image=image, command=command, args=container_args, env_vars=env_vars, ports=ports, predict_route=predict_route, health_route=health_route)
    if not autoscaling_hidden:
        self._ConfigureAutoScaling(version, min_nodes=min_nodes, max_nodes=max_nodes, metrics=metrics)
    for field_name, value in additional_fields.items():
        if value is not None:
            setattr(version, field_name, value)
    return version