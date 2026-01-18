import contextvars
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import ray
from ray.exceptions import RayActorError
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.common import ReplicaTag
from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE
from ray.serve.exceptions import RayServeException
from ray.serve.grpc_util import RayServegRPCContext
from ray.util.annotations import DeveloperAPI
def _get_global_client(_health_check_controller: bool=False, raise_if_no_controller_running: bool=True) -> Optional[ServeControllerClient]:
    """Gets the global client, which stores the controller's handle.

    Args:
        _health_check_controller: If True, run a health check on the
            cached controller if it exists. If the check fails, try reconnecting
            to the controller.
        raise_if_no_controller_running: Whether to raise an exception if
            there is no currently running Serve controller.

    Returns:
        ServeControllerClient to the running Serve controller. If there
        is no running controller and raise_if_no_controller_running is
        set to False, returns None.

    Raises:
        RayServeException: if there is no running Serve controller actor
        and raise_if_no_controller_running is set to True.
    """
    try:
        if _global_client is not None:
            if _health_check_controller:
                ray.get(_global_client._controller.check_alive.remote())
            return _global_client
    except RayActorError:
        logger.info('The cached controller has died. Reconnecting.')
        _set_global_client(None)
    return _connect(raise_if_no_controller_running)