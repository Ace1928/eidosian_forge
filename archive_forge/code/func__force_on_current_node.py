from typing import Dict, Optional, Union
import ray
def _force_on_current_node(remote_func_or_actor_class: Optional[Union[ray.remote_function.RemoteFunction, ray.actor.ActorClass]]=None) -> Union[Union[ray.remote_function.RemoteFunction, ray.actor.ActorClass], Dict]:
    """Schedule a remote function or actor class on the current node.

    If using Ray Client, the current node is the client server node.

    Args:
        remote_func_or_actor_class: A Ray remote function or actor class
            to schedule on the current node. If None, this function will directly
            return the options dict to pass to another remote function or actor class
            as remote options.
    Returns:
        The provided remote function or actor class, but with options modified to force
        placement on the current node. If remote_func_or_actor_class is None,
        the options dict to pass to another remote function or
        actor class as remote options kwargs.
    """
    current_node_id = ray.get_runtime_context().get_node_id()
    return _force_on_node(current_node_id, remote_func_or_actor_class)