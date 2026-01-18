import logging
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray import cloudpickle
from ray._private.utils import import_attr
from ray.exceptions import RuntimeEnvSetupError
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deploy_utils import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import DeploymentDetails, ServeApplicationSchema
from ray.types import ObjectRef
class ApplicationState:
    """Manage single application states with all operations"""

    def __init__(self, name: str, deployment_state_manager: DeploymentStateManager, endpoint_state: EndpointState, save_checkpoint_func: Callable):
        """
        Args:
            name: Application name.
            deployment_state_manager: State manager for all deployments
                in the cluster.
            endpoint_state: State manager for endpoints in the system.
            save_checkpoint_func: Function that can be called to write
                a checkpoint of the application state. This should be
                called in self._set_target_state() before actually
                setting the target state so that the controller can
                properly recover application states if it crashes.
        """
        self._name = name
        self._status_msg = ''
        self._deployment_state_manager = deployment_state_manager
        self._endpoint_state = endpoint_state
        self._route_prefix: Optional[str] = None
        self._docs_path: Optional[str] = None
        self._ingress_deployment_name: str = None
        self._status: ApplicationStatus = ApplicationStatus.DEPLOYING
        self._deployment_timestamp = time.time()
        self._build_app_task_info: Optional[BuildAppTaskInfo] = None
        self._target_state: ApplicationTargetState = ApplicationTargetState(deployment_infos=None, code_version=None, config=None, target_capacity=None, target_capacity_direction=None, deleting=False)
        self._save_checkpoint_func = save_checkpoint_func

    @property
    def route_prefix(self) -> Optional[str]:
        return self._route_prefix

    @property
    def docs_path(self) -> Optional[str]:
        return self._docs_path

    @property
    def status(self) -> ApplicationStatus:
        """Status of the application.

        DEPLOYING: The build task is still running, or the deployments
            have started deploying but aren't healthy yet.
        RUNNING: All deployments are healthy.
        DEPLOY_FAILED: The build task failed or one or more deployments
            became unhealthy in the process of deploying
        UNHEALTHY: While the application was running, one or more
            deployments transition from healthy to unhealthy.
        DELETING: Application and its deployments are being deleted.
        """
        return self._status

    @property
    def deployment_timestamp(self) -> int:
        return self._deployment_timestamp

    @property
    def target_deployments(self) -> List[str]:
        """List of target deployment names in application."""
        if self._target_state.deployment_infos is None:
            return []
        return list(self._target_state.deployment_infos.keys())

    @property
    def ingress_deployment(self) -> Optional[str]:
        return self._ingress_deployment_name

    def recover_target_state_from_checkpoint(self, checkpoint_data: ApplicationTargetState):
        logger.info(f"Recovering target state for application '{self._name}' from checkpoint.")
        self._set_target_state(checkpoint_data.deployment_infos, checkpoint_data.code_version, checkpoint_data.config, checkpoint_data.target_capacity, checkpoint_data.target_capacity_direction, checkpoint_data.deleting)

    def _set_target_state(self, deployment_infos: Optional[Dict[str, DeploymentInfo]], code_version: str, target_config: Optional[ServeApplicationSchema], target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None, deleting: bool=False):
        """Set application target state.

        While waiting for build task to finish, this should be
            (None, False)
        When build task has finished and during normal operation, this should be
            (target_deployments, False)
        When a request to delete the application has been received, this should be
            ({}, True)
        """
        if deleting:
            self._update_status(ApplicationStatus.DELETING)
        else:
            self._update_status(ApplicationStatus.DEPLOYING)
        if deployment_infos is None:
            self._ingress_deployment_name = None
        else:
            for name, info in deployment_infos.items():
                if info.ingress:
                    self._ingress_deployment_name = name
        target_state = ApplicationTargetState(deployment_infos, code_version, target_config, target_capacity, target_capacity_direction, deleting)
        self._save_checkpoint_func(writeahead_checkpoints={self._name: target_state})
        self._target_state = target_state

    def _set_target_state_deleting(self):
        """Set target state to deleting.

        Wipes the target deployment infos, code version, and config.
        """
        self._set_target_state(dict(), None, None, None, None, True)

    def _clear_target_state_and_store_config(self, target_config: Optional[ServeApplicationSchema]):
        """Clears the target state and stores the config."""
        self._set_target_state(None, None, target_config, None, None, False)

    def _delete_deployment(self, name):
        id = EndpointTag(name, self._name)
        self._endpoint_state.delete_endpoint(id)
        self._deployment_state_manager.delete_deployment(id)

    def delete(self):
        """Delete the application"""
        if self._status != ApplicationStatus.DELETING:
            logger.info(f"Deleting application '{self._name}'", extra={'log_to_stderr': False})
        self._set_target_state_deleting()

    def is_deleted(self) -> bool:
        """Check whether the application is already deleted.

        For an application to be considered deleted, the target state has to be set to
        deleting and all deployments have to be deleted.
        """
        return self._target_state.deleting and len(self._get_live_deployments()) == 0

    def apply_deployment_info(self, deployment_name: str, deployment_info: DeploymentInfo) -> None:
        """Deploys a deployment in the application."""
        route_prefix = deployment_info.route_prefix
        if route_prefix is not None and (not route_prefix.startswith('/')):
            raise RayServeException(f'Invalid route prefix "{route_prefix}", it must start with "/"')
        deployment_id = DeploymentID(deployment_name, self._name)
        self._deployment_state_manager.deploy(deployment_id, deployment_info)
        if deployment_info.route_prefix is not None:
            config = deployment_info.deployment_config
            self._endpoint_state.update_endpoint(deployment_id, EndpointInfo(route=deployment_info.route_prefix, app_is_cross_language=config.deployment_language != DeploymentLanguage.PYTHON))
        else:
            self._endpoint_state.delete_endpoint(deployment_id)

    def deploy(self, deployment_infos: Dict[str, DeploymentInfo]):
        """Deploy application from list of deployment infos.

        This function should only be called if the app is being deployed
        through serve.run instead of from a config.

        Raises: RayServeException if there is more than one route prefix
            or docs path.
        """
        self._route_prefix, self._docs_path = self._check_routes(deployment_infos)
        self._set_target_state(deployment_infos=deployment_infos, code_version=None, target_config=None, target_capacity=None, target_capacity_direction=None)

    def deploy_config(self, config: ServeApplicationSchema, target_capacity: Optional[float], target_capacity_direction: Optional[TargetCapacityDirection], deployment_time: int) -> None:
        """Deploys an application config.

        If the code version matches that of the current live deployments
        then it only applies the updated config to the deployment state
        manager. If the code version doesn't match, this will re-build
        the application.
        """
        self._deployment_timestamp = deployment_time
        config_version = get_app_code_version(config)
        if config_version == self._target_state.code_version:
            try:
                overrided_infos = override_deployment_info(self._name, self._target_state.deployment_infos, config)
                self._check_routes(overrided_infos)
                self._set_target_state(code_version=self._target_state.code_version, deployment_infos=overrided_infos, target_config=config, target_capacity=target_capacity, target_capacity_direction=target_capacity_direction)
            except (TypeError, ValueError, RayServeException):
                self._clear_target_state_and_store_config(config)
                self._update_status(ApplicationStatus.DEPLOY_FAILED, traceback.format_exc())
            except Exception:
                self._clear_target_state_and_store_config(config)
                self._update_status(ApplicationStatus.DEPLOY_FAILED, f"Unexpected error occured while applying config for application '{self._name}': \n{traceback.format_exc()}")
        else:
            if self._build_app_task_info and (not self._build_app_task_info.finished):
                logger.info(f"Received new config for application '{self._name}'. Cancelling previous request.")
                ray.cancel(self._build_app_task_info.obj_ref)
            self._clear_target_state_and_store_config(config)
            if self._target_state.config.runtime_env.get('container'):
                ServeUsageTag.APP_CONTAINER_RUNTIME_ENV_USED.record('1')
            logger.info(f"Building application '{self._name}'.")
            build_app_obj_ref = build_serve_application.options(runtime_env=config.runtime_env).remote(config.import_path, config.deployment_names, config_version, config.name, config.args)
            self._build_app_task_info = BuildAppTaskInfo(obj_ref=build_app_obj_ref, code_version=config_version, config=config, target_capacity=target_capacity, target_capacity_direction=target_capacity_direction, finished=False)

    def _get_live_deployments(self) -> List[str]:
        return self._deployment_state_manager.get_deployments_in_application(self._name)

    def _determine_app_status(self) -> Tuple[ApplicationStatus, str]:
        """Check deployment statuses and target state, and determine the
        corresponding application status.

        Returns:
            Status (ApplicationStatus):
                RUNNING: all deployments are healthy or autoscaling.
                DEPLOYING: there is one or more updating deployments,
                    and there are no unhealthy deployments.
                DEPLOY_FAILED: one or more deployments became unhealthy
                    while the application was deploying.
                UNHEALTHY: one or more deployments became unhealthy
                    while the application was running.
                DELETING: the application is being deleted.
            Error message (str):
                Non-empty string if status is DEPLOY_FAILED or UNHEALTHY
        """
        if self._target_state.deleting:
            return (ApplicationStatus.DELETING, '')
        num_healthy_deployments = 0
        num_autoscaling_deployments = 0
        num_updating_deployments = 0
        num_manually_scaling_deployments = 0
        unhealthy_deployment_names = []
        for deployment_status in self.get_deployments_statuses():
            if deployment_status.status == DeploymentStatus.UNHEALTHY:
                unhealthy_deployment_names.append(deployment_status.name)
            elif deployment_status.status == DeploymentStatus.HEALTHY:
                num_healthy_deployments += 1
            elif deployment_status.status_trigger == DeploymentStatusTrigger.AUTOSCALING:
                num_autoscaling_deployments += 1
            elif deployment_status.status == DeploymentStatus.UPDATING:
                num_updating_deployments += 1
            elif deployment_status.status in [DeploymentStatus.UPSCALING, DeploymentStatus.DOWNSCALING] and deployment_status.status_trigger == DeploymentStatusTrigger.CONFIG_UPDATE_STARTED:
                num_manually_scaling_deployments += 1
            else:
                raise RuntimeError(f'Found deployment with unexpected status {deployment_status.status} and status trigger {deployment_status.status_trigger}.')
        if len(unhealthy_deployment_names):
            status_msg = f'The deployments {unhealthy_deployment_names} are UNHEALTHY.'
            if self._status in [ApplicationStatus.DEPLOYING, ApplicationStatus.DEPLOY_FAILED]:
                return (ApplicationStatus.DEPLOY_FAILED, status_msg)
            else:
                return (ApplicationStatus.UNHEALTHY, status_msg)
        elif num_updating_deployments + num_manually_scaling_deployments > 0:
            return (ApplicationStatus.DEPLOYING, '')
        else:
            assert num_healthy_deployments + num_autoscaling_deployments == len(self.target_deployments)
            return (ApplicationStatus.RUNNING, '')

    def _reconcile_build_app_task(self) -> Tuple[Tuple, BuildAppStatus, str]:
        """If necessary, reconcile the in-progress build task.

        Returns:
            Deploy arguments (Dict[str, DeploymentInfo]):
                The deploy arguments returned from the build app task
                and their code version.
            Status (BuildAppStatus):
                NO_TASK_IN_PROGRESS: There is no build task to reconcile.
                SUCCEEDED: Task finished successfully.
                FAILED: An error occurred during execution of build app task
                IN_PROGRESS: Task hasn't finished yet.
            Error message (str):
                Non-empty string if status is DEPLOY_FAILED or UNHEALTHY
        """
        if self._build_app_task_info is None or self._build_app_task_info.finished:
            return (None, BuildAppStatus.NO_TASK_IN_PROGRESS, '')
        if not check_obj_ref_ready_nowait(self._build_app_task_info.obj_ref):
            return (None, BuildAppStatus.IN_PROGRESS, '')
        self._build_app_task_info.finished = True
        try:
            args, err = ray.get(self._build_app_task_info.obj_ref)
            if err is None:
                logger.info(f"Built application '{self._name}' successfully.")
            else:
                return (None, BuildAppStatus.FAILED, f"Deploying app '{self._name}' failed with exception:\n{err}")
        except RuntimeEnvSetupError:
            error_msg = f"Runtime env setup for app '{self._name}' failed:\n" + traceback.format_exc()
            return (None, BuildAppStatus.FAILED, error_msg)
        except Exception:
            error_msg = f"Unexpected error occured while deploying application '{self._name}': \n{traceback.format_exc()}"
            return (None, BuildAppStatus.FAILED, error_msg)
        try:
            deployment_infos = {params['deployment_name']: deploy_args_to_deployment_info(**params, app_name=self._name) for params in args}
            overrided_infos = override_deployment_info(self._name, deployment_infos, self._build_app_task_info.config)
            self._route_prefix, self._docs_path = self._check_routes(overrided_infos)
            return (overrided_infos, BuildAppStatus.SUCCEEDED, '')
        except (TypeError, ValueError, RayServeException):
            return (None, BuildAppStatus.FAILED, traceback.format_exc())
        except Exception:
            error_msg = f"Unexpected error occured while applying config for application '{self._name}': \n{traceback.format_exc()}"
            return (None, BuildAppStatus.FAILED, error_msg)

    def _check_routes(self, deployment_infos: Dict[str, DeploymentInfo]) -> Tuple[str, str]:
        """Check route prefixes and docs paths of deployments in app.

        There should only be one non-null route prefix. If there is one,
        set it as the application route prefix. This function must be
        run every control loop iteration because the target config could
        be updated without kicking off a new task.

        Returns: tuple of route prefix, docs path.
        Raises: RayServeException if more than one route prefix or docs
            path is found among deployments.
        """
        num_route_prefixes = 0
        num_docs_paths = 0
        route_prefix = None
        docs_path = None
        for info in deployment_infos.values():
            if info.route_prefix is not None:
                route_prefix = info.route_prefix
                num_route_prefixes += 1
            if info.docs_path is not None:
                docs_path = info.docs_path
                num_docs_paths += 1
        if num_route_prefixes > 1:
            raise RayServeException(f'Found multiple route prefixes from application "{self._name}", Please specify only one route prefix for the application to avoid this issue.')
        if num_docs_paths > 1:
            raise RayServeException(f'Found multiple deployments in application "{self._name}" that have a docs path. This may be due to using multiple FastAPI deployments in your application. Please only include one deployment with a docs path in your application to avoid this issue.')
        return (route_prefix, docs_path)

    def _reconcile_target_deployments(self) -> None:
        """Reconcile target deployments in application target state.

        Ensure each deployment is running on up-to-date info, and
        remove outdated deployments from the application.
        """
        for deployment_name, info in self._target_state.deployment_infos.items():
            deploy_info = deepcopy(info)
            deploy_info.set_target_capacity(new_target_capacity=self._target_state.target_capacity, new_target_capacity_direction=self._target_state.target_capacity_direction)
            if self._target_state.config and self._target_state.config.logging_config and (deploy_info.deployment_config.logging_config is None):
                deploy_info.deployment_config.logging_config = self._target_state.config.logging_config
            self.apply_deployment_info(deployment_name, deploy_info)
        for deployment_name in self._get_live_deployments():
            if deployment_name not in self.target_deployments:
                self._delete_deployment(deployment_name)

    def update(self) -> bool:
        """Attempts to reconcile this application to match its target state.

        Updates the application status and status message based on the
        current state of the system.

        Returns:
            A boolean indicating whether the application is ready to be
            deleted.
        """
        infos, task_status, msg = self._reconcile_build_app_task()
        if task_status == BuildAppStatus.SUCCEEDED:
            self._set_target_state(deployment_infos=infos, code_version=self._build_app_task_info.code_version, target_config=self._build_app_task_info.config, target_capacity=self._build_app_task_info.target_capacity, target_capacity_direction=self._build_app_task_info.target_capacity_direction)
        elif task_status == BuildAppStatus.FAILED:
            self._update_status(ApplicationStatus.DEPLOY_FAILED, msg)
        if self._target_state.deployment_infos is not None:
            self._reconcile_target_deployments()
            status, status_msg = self._determine_app_status()
            self._update_status(status, status_msg)
        if self._target_state.deleting:
            return self.is_deleted()
        return False

    def get_checkpoint_data(self) -> ApplicationTargetState:
        return self._target_state

    def get_deployments_statuses(self) -> List[DeploymentStatusInfo]:
        """Return all deployment status information"""
        deployments = [DeploymentID(deployment, self._name) for deployment in self.target_deployments]
        return self._deployment_state_manager.get_deployment_statuses(deployments)

    def get_application_status_info(self) -> ApplicationStatusInfo:
        """Return the application status information"""
        return ApplicationStatusInfo(self._status, message=self._status_msg, deployment_timestamp=self._deployment_timestamp)

    def list_deployment_details(self) -> Dict[str, DeploymentDetails]:
        """Gets detailed info on all live deployments in this application.
        (Does not include deleted deployments.)

        Returns:
            A dictionary of deployment infos. The set of deployment info returned
            may not be the full list of deployments that are part of the application.
            This can happen when the application is still deploying and bringing up
            deployments, or when the application is deleting and some deployments have
            been deleted.
        """
        details = {deployment_name: self._deployment_state_manager.get_deployment_details(DeploymentID(deployment_name, self._name)) for deployment_name in self.target_deployments}
        return {k: v for k, v in details.items() if v is not None}

    def _update_status(self, status: ApplicationStatus, status_msg: str='') -> None:
        if status_msg and status in [ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.UNHEALTHY]:
            logger.warning(status_msg)
        self._status = status
        self._status_msg = status_msg