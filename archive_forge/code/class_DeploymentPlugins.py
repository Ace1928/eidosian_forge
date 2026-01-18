import abc
import inspect
import entrypoints
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.utils import parse_target_uri
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.annotations import developer_stable
@developer_stable
class DeploymentPlugins(PluginManager):

    def __init__(self):
        super().__init__('mlflow.deployments')
        self.register_entrypoints()

    def __getitem__(self, item):
        """Override __getitem__ so that we can directly look up plugins via dict-like syntax"""
        try:
            target_name = parse_target_uri(item)
            plugin_like = self.registry[target_name]
        except KeyError:
            msg = f'No plugin found for managing model deployments to "{item}". In order to deploy models to "{item}", find and install an appropriate plugin from https://mlflow.org/docs/latest/plugins.html#community-plugins using your package manager (pip, conda etc).'
            raise MlflowException(msg, error_code=RESOURCE_DOES_NOT_EXIST)
        if isinstance(plugin_like, entrypoints.EntryPoint):
            try:
                plugin_obj = plugin_like.load()
            except (AttributeError, ImportError) as exc:
                raise RuntimeError(f'Failed to load the plugin "{item}": {exc}')
            self.registry[item] = plugin_obj
        else:
            plugin_obj = plugin_like
        expected = {'target_help', 'run_local'}
        deployment_classes = []
        for name, obj in inspect.getmembers(plugin_obj):
            if name in expected:
                expected.remove(name)
            elif inspect.isclass(obj) and issubclass(obj, BaseDeploymentClient) and (not obj == BaseDeploymentClient):
                deployment_classes.append(name)
        if len(expected) > 0:
            raise MlflowException(f'Plugin registered for the target {item} does not have all the required interfaces. Raise an issue with the plugin developers.\nMissing interfaces: {expected}', error_code=INTERNAL_ERROR)
        if len(deployment_classes) > 1:
            raise MlflowException(f'Plugin registered for the target {item} has more than one child class of BaseDeploymentClient. Raise an issue with the plugin developers. Classes found are {deployment_classes}')
        elif len(deployment_classes) == 0:
            raise MlflowException(f'Plugin registered for the target {item} has no child class of BaseDeploymentClient. Raise an issue with the plugin developers')
        return plugin_obj