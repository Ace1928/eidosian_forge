import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from ray import cloudpickle
from ray._private import ray_option_utils
from ray._private.protobuf_compat import message_to_dict
from ray._private.pydantic_compat import (
from ray._private.serialization import pickle_dumps
from ray._private.utils import resources_from_ray_options
from ray.serve._private.constants import (
from ray.serve._private.utils import DEFAULT, DeploymentOptionUpdateType
from ray.serve.config import AutoscalingConfig
from ray.serve.generated.serve_pb2 import AutoscalingConfig as AutoscalingConfigProto
from ray.serve.generated.serve_pb2 import DeploymentConfig as DeploymentConfigProto
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.generated.serve_pb2 import EncodingType as EncodingTypeProto
from ray.serve.generated.serve_pb2 import LoggingConfig as LoggingConfigProto
from ray.serve.generated.serve_pb2 import ReplicaConfig as ReplicaConfigProto
from ray.util.placement_group import VALID_PLACEMENT_GROUP_STRATEGIES
class DeploymentConfig(BaseModel):
    """Internal datastructure wrapping config options for a deployment.

    Args:
        num_replicas (Optional[int]): The number of processes to start up that
            handles requests to this deployment. Defaults to 1.
        max_concurrent_queries (Optional[int]): The maximum number of queries
            that is sent to a replica of this deployment without receiving
            a response. Defaults to 100.
        user_config (Optional[Any]): Arguments to pass to the reconfigure
            method of the deployment. The reconfigure method is called if
            user_config is not None. Must be JSON-serializable.
        graceful_shutdown_wait_loop_s (Optional[float]): Duration
            that deployment replicas wait until there is no more work to
            be done before shutting down.
        graceful_shutdown_timeout_s (Optional[float]):
            Controller waits for this duration to forcefully kill the replica
            for shutdown.
        health_check_period_s (Optional[float]):
            Frequency at which the controller health checks replicas.
        health_check_timeout_s (Optional[float]):
            Timeout that the controller waits for a response from the
            replica's health check before marking it unhealthy.
        user_configured_option_names (Set[str]):
            The names of options manually configured by the user.
    """
    num_replicas: NonNegativeInt = Field(default=1, update_type=DeploymentOptionUpdateType.LightWeight)
    max_concurrent_queries: Optional[int] = Field(default=None, update_type=DeploymentOptionUpdateType.NeedsReconfigure)
    user_config: Any = Field(default=None, update_type=DeploymentOptionUpdateType.NeedsActorReconfigure)
    graceful_shutdown_timeout_s: NonNegativeFloat = Field(default=DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_S, update_type=DeploymentOptionUpdateType.NeedsReconfigure)
    graceful_shutdown_wait_loop_s: NonNegativeFloat = Field(default=DEFAULT_GRACEFUL_SHUTDOWN_WAIT_LOOP_S, update_type=DeploymentOptionUpdateType.NeedsActorReconfigure)
    health_check_period_s: PositiveFloat = Field(default=DEFAULT_HEALTH_CHECK_PERIOD_S, update_type=DeploymentOptionUpdateType.NeedsReconfigure)
    health_check_timeout_s: PositiveFloat = Field(default=DEFAULT_HEALTH_CHECK_TIMEOUT_S, update_type=DeploymentOptionUpdateType.NeedsReconfigure)
    autoscaling_config: Optional[AutoscalingConfig] = Field(default=None, update_type=DeploymentOptionUpdateType.LightWeight)
    is_cross_language: bool = False
    deployment_language: Any = DeploymentLanguage.PYTHON
    version: Optional[str] = Field(default=None, update_type=DeploymentOptionUpdateType.HeavyWeight)
    logging_config: Optional[dict] = Field(default=None, update_type=DeploymentOptionUpdateType.NeedsActorReconfigure)
    user_configured_option_names: Set[str] = set()

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    @validator('max_concurrent_queries', always=True)
    def set_max_queries_by_mode(cls, v, values):
        if v is None:
            v = DEFAULT_MAX_CONCURRENT_QUERIES
        elif v <= 0:
            raise ValueError('max_concurrent_queries must be >= 0')
        return v

    @validator('user_config', always=True)
    def user_config_json_serializable(cls, v):
        if isinstance(v, bytes):
            return v
        if v is not None:
            try:
                json.dumps(v)
            except TypeError as e:
                raise ValueError(f'user_config is not JSON-serializable: {str(e)}.')
        return v

    @validator('logging_config', always=True)
    def logging_config_valid(cls, v):
        if v is None:
            return v
        if not isinstance(v, dict):
            raise TypeError(f"Got invalid type '{type(v)}' for logging_config. Expected a dictionary.")
        from ray.serve.schema import LoggingConfig
        v = LoggingConfig(**v).dict()
        return v

    def needs_pickle(self):
        return _needs_pickle(self.deployment_language, self.is_cross_language)

    def to_proto(self):
        data = self.dict()
        if data.get('user_config') is not None:
            if self.needs_pickle():
                data['user_config'] = cloudpickle.dumps(data['user_config'])
        if data.get('autoscaling_config'):
            data['autoscaling_config'] = AutoscalingConfigProto(**data['autoscaling_config'])
        if data.get('logging_config'):
            if 'encoding' in data['logging_config']:
                data['logging_config']['encoding'] = EncodingTypeProto.Value(data['logging_config']['encoding'])
            data['logging_config'] = LoggingConfigProto(**data['logging_config'])
        data['user_configured_option_names'] = list(data['user_configured_option_names'])
        return DeploymentConfigProto(**data)

    def to_proto_bytes(self):
        return self.to_proto().SerializeToString()

    @classmethod
    def from_proto(cls, proto: DeploymentConfigProto):
        data = message_to_dict(proto, always_print_fields_with_no_presence=True, preserving_proto_field_name=True, use_integers_for_enums=True)
        if 'user_config' in data:
            if data['user_config'] != '':
                deployment_language = data['deployment_language'] if 'deployment_language' in data else DeploymentLanguage.PYTHON
                is_cross_language = data['is_cross_language'] if 'is_cross_language' in data else False
                needs_pickle = _needs_pickle(deployment_language, is_cross_language)
                if needs_pickle:
                    data['user_config'] = cloudpickle.loads(proto.user_config)
                else:
                    data['user_config'] = proto.user_config
            else:
                data['user_config'] = None
        if 'autoscaling_config' in data:
            if not data['autoscaling_config'].get('upscale_smoothing_factor'):
                data['autoscaling_config']['upscale_smoothing_factor'] = None
            if not data['autoscaling_config'].get('downscale_smoothing_factor'):
                data['autoscaling_config']['downscale_smoothing_factor'] = None
            data['autoscaling_config'] = AutoscalingConfig(**data['autoscaling_config'])
        if 'version' in data:
            if data['version'] == '':
                data['version'] = None
        if 'user_configured_option_names' in data:
            data['user_configured_option_names'] = set(data['user_configured_option_names'])
        if 'logging_config' in data:
            if 'encoding' in data['logging_config']:
                data['logging_config']['encoding'] = EncodingTypeProto.Name(data['logging_config']['encoding'])
        return cls(**data)

    @classmethod
    def from_proto_bytes(cls, proto_bytes: bytes):
        proto = DeploymentConfigProto.FromString(proto_bytes)
        return cls.from_proto(proto)

    @classmethod
    def from_default(cls, **kwargs):
        """Creates a default DeploymentConfig and overrides it with kwargs.

        Ignores any kwargs set to DEFAULT.VALUE.

        Raises:
            TypeError: when a keyword that's not an argument to the class is
                passed in.
        """
        config = cls()
        valid_config_options = set(config.dict().keys())
        for key, val in kwargs.items():
            if key not in valid_config_options:
                raise TypeError(f'Got invalid Deployment config option "{key}" (with value {val}) as keyword argument. All Deployment config options must come from this list: {list(valid_config_options)}.')
        kwargs = {key: val for key, val in kwargs.items() if val != DEFAULT.VALUE}
        for key, val in kwargs.items():
            config.__setattr__(key, val)
        return config