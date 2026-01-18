from typing import Dict, Optional, Union
import ray
def _force_on_node(node_id: str, remote_func_or_actor_class: Optional[Union[ray.remote_function.RemoteFunction, ray.actor.ActorClass]]=None) -> Union[Union[ray.remote_function.RemoteFunction, ray.actor.ActorClass], Dict]:
    """Schedule a remote function or actor class on a given node.

    Args:
        node_id: The node to schedule on.
        remote_func_or_actor_class: A Ray remote function or actor class
            to schedule on the input node. If None, this function will directly
            return the options dict to pass to another remote function or actor class
            as remote options.
    Returns:
        The provided remote function or actor class, but with options modified to force
        placement on the input node. If remote_func_or_actor_class is None,
        the options dict to pass to another remote function or
        actor class as remote options kwargs.
    """
    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    options = {'scheduling_strategy': scheduling_strategy}
    if remote_func_or_actor_class is None:
        return options
    return remote_func_or_actor_class.options(**options)