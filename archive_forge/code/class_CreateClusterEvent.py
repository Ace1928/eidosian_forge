from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from ray.autoscaler._private.cli_logger import cli_logger
class CreateClusterEvent(Enum):
    """Events to track in ray.autoscaler.sdk.create_or_update_cluster.

    Attributes:
        up_started : Invoked at the beginning of create_or_update_cluster.
        ssh_keypair_downloaded : Invoked when the ssh keypair is downloaded.
        cluster_booting_started : Invoked when when the cluster booting starts.
        acquiring_new_head_node : Invoked before the head node is acquired.
        head_node_acquired : Invoked after the head node is acquired.
        ssh_control_acquired : Invoked when the node is being updated.
        run_initialization_cmd : Invoked before all initialization
            commands are called and again before each initialization command.
        run_setup_cmd : Invoked before all setup commands are
            called and again before each setup command.
        start_ray_runtime : Invoked before ray start commands are run.
        start_ray_runtime_completed : Invoked after ray start commands
            are run.
        cluster_booting_completed : Invoked after cluster booting
            is completed.
    """
    up_started = auto()
    ssh_keypair_downloaded = auto()
    cluster_booting_started = auto()
    acquiring_new_head_node = auto()
    head_node_acquired = auto()
    ssh_control_acquired = auto()
    run_initialization_cmd = auto()
    run_setup_cmd = auto()
    start_ray_runtime = auto()
    start_ray_runtime_completed = auto()
    cluster_booting_completed = auto()