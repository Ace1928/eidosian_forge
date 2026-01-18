import inspect
from logging import Logger
from mlflow.deployments.base import BaseDeploymentClient
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.deployments.utils import get_deployments_target, parse_target_uri
from mlflow.exceptions import MlflowException
def _target_help(target):
    """
    Return a string containing detailed documentation on the current deployment target,
    to be displayed when users invoke the ``mlflow deployments help -t <target-name>`` CLI.
    This method should be defined within the module specified by the plugin author.
    The string should contain:
    * An explanation of target-specific fields in the ``config`` passed to ``create_deployment``,
      ``update_deployment``
    * How to specify a ``target_uri`` (e.g. for AWS SageMaker, ``target_uri``s have a scheme of
      "sagemaker:/<aws-cli-profile-name>", where aws-cli-profile-name is the name of an AWS
      CLI profile https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)
    * Any other target-specific details.

    Args:
        target: Which target to use. This information is used to call the appropriate plugin.
    """
    return plugin_store[target].target_help()