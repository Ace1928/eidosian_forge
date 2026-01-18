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
class ReplicaConfig:
    """Internal datastructure wrapping config options for a deployment's replicas.

    Provides five main properties (see property docstrings for more info):
        deployment_def: the code, or a reference to the code, that this
            replica should run.
        init_args: the deployment_def's init_args.
        init_kwargs: the deployment_def's init_kwargs.
        ray_actor_options: the Ray actor options to pass into the replica's
            actor.
        resource_dict: contains info on this replica's actor's resource needs.

    Offers a serialized equivalent (e.g. serialized_deployment_def) for
    deployment_def, init_args, and init_kwargs. Deserializes these properties
    when they're first accessed, if they were not passed in directly through
    create().

    Use the classmethod create() to make a ReplicaConfig with the deserialized
    properties.

    Note: overwriting or setting any property after the ReplicaConfig has been
    constructed is currently undefined behavior. The config's fields should not
    be modified externally after it is created.
    """

    def __init__(self, deployment_def_name: str, serialized_deployment_def: bytes, serialized_init_args: bytes, serialized_init_kwargs: bytes, ray_actor_options: Dict, placement_group_bundles: Optional[List[Dict[str, float]]]=None, placement_group_strategy: Optional[str]=None, max_replicas_per_node: Optional[int]=None, needs_pickle: bool=True):
        """Construct a ReplicaConfig with serialized properties.

        All parameters are required. See classmethod create() for defaults.
        """
        self.deployment_def_name = deployment_def_name
        self.serialized_deployment_def = serialized_deployment_def
        self.serialized_init_args = serialized_init_args
        self.serialized_init_kwargs = serialized_init_kwargs
        self._deployment_def = None
        self._init_args = None
        self._init_kwargs = None
        self.ray_actor_options = ray_actor_options
        self._validate_ray_actor_options()
        self.placement_group_bundles = placement_group_bundles
        self.placement_group_strategy = placement_group_strategy
        self._validate_placement_group_options()
        self.max_replicas_per_node = max_replicas_per_node
        self._validate_max_replicas_per_node()
        self.resource_dict = resources_from_ray_options(self.ray_actor_options)
        self.needs_pickle = needs_pickle

    def update_ray_actor_options(self, ray_actor_options):
        self.ray_actor_options = ray_actor_options
        self._validate_ray_actor_options()
        self.resource_dict = resources_from_ray_options(self.ray_actor_options)

    def update_placement_group_options(self, placement_group_bundles: Optional[List[Dict[str, float]]], placement_group_strategy: Optional[str]):
        self.placement_group_bundles = placement_group_bundles
        self.placement_group_strategy = placement_group_strategy
        self._validate_placement_group_options()

    def update_max_replicas_per_node(self, max_replicas_per_node: Optional[int]):
        self.max_replicas_per_node = max_replicas_per_node
        self._validate_max_replicas_per_node()

    @classmethod
    def create(cls, deployment_def: Union[Callable, str], init_args: Optional[Tuple[Any]]=None, init_kwargs: Optional[Dict[Any, Any]]=None, ray_actor_options: Optional[Dict]=None, placement_group_bundles: Optional[List[Dict[str, float]]]=None, placement_group_strategy: Optional[str]=None, max_replicas_per_node: Optional[int]=None, deployment_def_name: Optional[str]=None):
        """Create a ReplicaConfig from deserialized parameters."""
        if not callable(deployment_def) and (not isinstance(deployment_def, str)):
            raise TypeError('@serve.deployment must be called on a class or function.')
        if not (init_args is None or isinstance(init_args, (tuple, list))):
            raise TypeError('init_args must be a tuple.')
        if not (init_kwargs is None or isinstance(init_kwargs, dict)):
            raise TypeError('init_kwargs must be a dict.')
        if inspect.isfunction(deployment_def):
            if init_args:
                raise ValueError('init_args not supported for function deployments.')
            elif init_kwargs:
                raise ValueError('init_kwargs not supported for function deployments.')
        if not isinstance(deployment_def, (Callable, str)):
            raise TypeError(f'Got invalid type "{type(deployment_def)}" for deployment_def. Expected deployment_def to be a class, function, or string.')
        if init_args is None:
            init_args = ()
        if init_kwargs is None:
            init_kwargs = {}
        if ray_actor_options is None:
            ray_actor_options = {}
        if deployment_def_name is None:
            if isinstance(deployment_def, str):
                deployment_def_name = deployment_def
            else:
                deployment_def_name = deployment_def.__name__
        config = cls(deployment_def_name, pickle_dumps(deployment_def, f'Could not serialize the deployment {repr(deployment_def)}'), pickle_dumps(init_args, 'Could not serialize the deployment init args'), pickle_dumps(init_kwargs, 'Could not serialize the deployment init kwargs'), ray_actor_options, placement_group_bundles, placement_group_strategy, max_replicas_per_node)
        config._deployment_def = deployment_def
        config._init_args = init_args
        config._init_kwargs = init_kwargs
        return config

    def _validate_ray_actor_options(self):
        if not isinstance(self.ray_actor_options, dict):
            raise TypeError(f'Got invalid type "{type(self.ray_actor_options)}" for ray_actor_options. Expected a dictionary.')
        allowed_ray_actor_options = {'accelerator_type', 'memory', 'num_cpus', 'num_gpus', 'object_store_memory', 'resources', 'runtime_env'}
        for option in self.ray_actor_options:
            if option not in allowed_ray_actor_options:
                raise ValueError(f"Specifying '{option}' in ray_actor_options is not allowed. Allowed options: {allowed_ray_actor_options}")
        ray_option_utils.validate_actor_options(self.ray_actor_options, in_options=True)
        if self.ray_actor_options.get('num_cpus') is None:
            self.ray_actor_options['num_cpus'] = 1

    def _validate_max_replicas_per_node(self) -> None:
        if self.max_replicas_per_node is None:
            return
        if not isinstance(self.max_replicas_per_node, int):
            raise TypeError(f"Get invalid type '{type(self.max_replicas_per_node)}' for max_replicas_per_node. Expected None or an integer in the range of [1, {MAX_REPLICAS_PER_NODE_MAX_VALUE}].")
        if self.max_replicas_per_node < 1 or self.max_replicas_per_node > MAX_REPLICAS_PER_NODE_MAX_VALUE:
            raise ValueError(f'Invalid max_replicas_per_node {self.max_replicas_per_node}. Valid values are None or an integer in the range of [1, {MAX_REPLICAS_PER_NODE_MAX_VALUE}].')

    def _validate_placement_group_options(self) -> None:
        if self.placement_group_strategy is not None and self.placement_group_strategy not in VALID_PLACEMENT_GROUP_STRATEGIES:
            raise ValueError(f"Invalid placement group strategy '{self.placement_group_strategy}'. Supported strategies are: {VALID_PLACEMENT_GROUP_STRATEGIES}.")
        if self.placement_group_strategy is not None and self.placement_group_bundles is None:
            raise ValueError('If `placement_group_strategy` is provided, `placement_group_bundles` must also be provided.')
        if self.placement_group_bundles is not None:
            if not isinstance(self.placement_group_bundles, list) or len(self.placement_group_bundles) == 0:
                raise ValueError('`placement_group_bundles` must be a non-empty list of resource dictionaries. For example: `[{"CPU": 1.0}, {"GPU": 1.0}]`.')
            for i, bundle in enumerate(self.placement_group_bundles):
                if not isinstance(bundle, dict) or not all((isinstance(k, str) for k in bundle.keys())) or (not all((isinstance(v, (int, float)) for v in bundle.values()))):
                    raise ValueError('`placement_group_bundles` must be a non-empty list of resource dictionaries. For example: `[{"CPU": 1.0}, {"GPU": 1.0}]`.')
                if i == 0:
                    bundle_cpu = bundle.get('CPU', 0)
                    replica_actor_num_cpus = self.ray_actor_options.get('num_cpus', 0)
                    if bundle_cpu < replica_actor_num_cpus:
                        raise ValueError(f'When using `placement_group_bundles`, the replica actor will be placed in the first bundle, so the resource requirements for the actor must be a subset of the first bundle. `num_cpus` for the actor is {replica_actor_num_cpus} but the bundle only has {bundle_cpu} `CPU` specified.')
                    bundle_gpu = bundle.get('GPU', 0)
                    replica_actor_num_gpus = self.ray_actor_options.get('num_gpus', 0)
                    if bundle_gpu < replica_actor_num_gpus:
                        raise ValueError(f'When using `placement_group_bundles`, the replica actor will be placed in the first bundle, so the resource requirements for the actor must be a subset of the first bundle. `num_gpus` for the actor is {replica_actor_num_gpus} but the bundle only has {bundle_gpu} `GPU` specified.')
                    replica_actor_resources = self.ray_actor_options.get('resources', {})
                    for actor_resource, actor_value in replica_actor_resources.items():
                        bundle_value = bundle.get(actor_resource, 0)
                        if bundle_value < actor_value:
                            raise ValueError(f'When using `placement_group_bundles`, the replica actor will be placed in the first bundle, so the resource requirements for the actor must be a subset of the first bundle. `{actor_resource}` requirement for the actor is {actor_value} but the bundle only has {bundle_value} `{actor_resource}` specified.')

    @property
    def deployment_def(self) -> Union[Callable, str]:
        """The code, or a reference to the code, that this replica runs.

        For Python replicas, this can be one of the following:
            - Function (Callable)
            - Class (Callable)
            - Import path (str)

        For Java replicas, this can be one of the following:
            - Class path (str)
        """
        if self._deployment_def is None:
            if self.needs_pickle:
                self._deployment_def = cloudpickle.loads(self.serialized_deployment_def)
            else:
                self._deployment_def = self.serialized_deployment_def.decode(encoding='utf-8')
        return self._deployment_def

    @property
    def init_args(self) -> Optional[Union[Tuple[Any], bytes]]:
        """The init_args for a Python class.

        This property is only meaningful if deployment_def is a Python class.
        Otherwise, it is None.
        """
        if self._init_args is None:
            if self.needs_pickle:
                self._init_args = cloudpickle.loads(self.serialized_init_args)
            else:
                self._init_args = self.serialized_init_args
        return self._init_args

    @property
    def init_kwargs(self) -> Optional[Tuple[Any]]:
        """The init_kwargs for a Python class.

        This property is only meaningful if deployment_def is a Python class.
        Otherwise, it is None.
        """
        if self._init_kwargs is None:
            self._init_kwargs = cloudpickle.loads(self.serialized_init_kwargs)
        return self._init_kwargs

    @classmethod
    def from_proto(cls, proto: ReplicaConfigProto, needs_pickle: bool=True):
        return ReplicaConfig(proto.deployment_def_name, proto.deployment_def, proto.init_args if proto.init_args != b'' else None, proto.init_kwargs if proto.init_kwargs != b'' else None, json.loads(proto.ray_actor_options), json.loads(proto.placement_group_bundles) if proto.placement_group_bundles else None, proto.placement_group_strategy if proto.placement_group_strategy != '' else None, proto.max_replicas_per_node if proto.max_replicas_per_node else None, needs_pickle)

    @classmethod
    def from_proto_bytes(cls, proto_bytes: bytes, needs_pickle: bool=True):
        proto = ReplicaConfigProto.FromString(proto_bytes)
        return cls.from_proto(proto, needs_pickle)

    def to_proto(self):
        return ReplicaConfigProto(deployment_def_name=self.deployment_def_name, deployment_def=self.serialized_deployment_def, init_args=self.serialized_init_args, init_kwargs=self.serialized_init_kwargs, ray_actor_options=json.dumps(self.ray_actor_options), placement_group_bundles=json.dumps(self.placement_group_bundles) if self.placement_group_bundles is not None else '', placement_group_strategy=self.placement_group_strategy, max_replicas_per_node=self.max_replicas_per_node if self.max_replicas_per_node is not None else 0)

    def to_proto_bytes(self):
        return self.to_proto().SerializeToString()