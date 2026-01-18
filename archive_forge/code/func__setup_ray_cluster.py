import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def _setup_ray_cluster(*, num_worker_nodes: int, num_cpus_worker_node: int, num_cpus_head_node: int, num_gpus_worker_node: int, num_gpus_head_node: int, using_stage_scheduling: bool, heap_memory_worker_node: int, heap_memory_head_node: int, object_store_memory_worker_node: int, object_store_memory_head_node: int, head_node_options: Dict, worker_node_options: Dict, ray_temp_root_dir: str, collect_log_to_path: str, autoscale: bool, autoscale_upscaling_speed: float, autoscale_idle_timeout_minutes: float) -> Type[RayClusterOnSpark]:
    """
    The public API `ray.util.spark.setup_ray_cluster` does some argument
    validation and then pass validated arguments to this interface.
    and it returns a `RayClusterOnSpark` instance.

    The returned instance can be used to connect to, disconnect from and shutdown the
    ray cluster. This instance can also be used as a context manager (used by
    encapsulating operations within `with _setup_ray_cluster(...):`). Upon entering the
    managed scope, the ray cluster is initiated and connected to. When exiting the
    scope, the ray cluster is disconnected and shut down.

    Note: This function interface is stable and can be used for
    instrumentation logging patching.
    """
    from pyspark.util import inheritable_thread_target
    if RAY_ON_SPARK_START_HOOK in os.environ:
        start_hook = _load_class(os.environ[RAY_ON_SPARK_START_HOOK])()
    elif is_in_databricks_runtime():
        start_hook = DefaultDatabricksRayOnSparkStartHook()
    else:
        start_hook = RayOnSparkStartHook()
    spark = get_spark_session()
    ray_head_ip = socket.gethostbyname(get_spark_application_driver_host(spark))
    ray_head_port = get_random_unused_port(ray_head_ip, min_port=9000, max_port=10000)
    port_exclude_list = [ray_head_port]
    head_node_options = head_node_options.copy()
    include_dashboard = head_node_options.pop('include_dashboard', None)
    ray_dashboard_port = head_node_options.pop('dashboard_port', None)
    ray_client_server_port = get_random_unused_port(ray_head_ip, min_port=9000, max_port=10000, exclude_list=port_exclude_list)
    port_exclude_list.append(ray_client_server_port)
    if autoscale:
        spark_job_server_port = get_random_unused_port(ray_head_ip, min_port=9000, max_port=10000, exclude_list=port_exclude_list)
        port_exclude_list.append(spark_job_server_port)
    else:
        spark_job_server_port = None
    if include_dashboard is None or include_dashboard is True:
        if ray_dashboard_port is None:
            ray_dashboard_port = get_random_unused_port(ray_head_ip, min_port=9000, max_port=10000, exclude_list=port_exclude_list)
            port_exclude_list.append(ray_dashboard_port)
        ray_dashboard_agent_port = get_random_unused_port(ray_head_ip, min_port=9000, max_port=10000, exclude_list=port_exclude_list)
        port_exclude_list.append(ray_dashboard_agent_port)
        dashboard_options = ['--dashboard-host=0.0.0.0', f'--dashboard-port={ray_dashboard_port}', f'--dashboard-agent-listen-port={ray_dashboard_agent_port}']
        if include_dashboard is True:
            dashboard_options += ['--include-dashboard=true']
    else:
        dashboard_options = ['--include-dashboard=false']
    _logger.info(f'Ray head hostname: {ray_head_ip}, port: {ray_head_port}, ray client server port: {ray_client_server_port}.')
    cluster_unique_id = uuid.uuid4().hex[:8]
    if ray_temp_root_dir is None:
        ray_temp_root_dir = start_hook.get_default_temp_dir()
    ray_temp_dir = os.path.join(ray_temp_root_dir, f'ray-{ray_head_port}-{cluster_unique_id}')
    os.makedirs(ray_temp_dir, exist_ok=True)
    object_spilling_dir = os.path.join(ray_temp_dir, 'spill')
    os.makedirs(object_spilling_dir, exist_ok=True)
    head_node_options = _append_default_spilling_dir_config(head_node_options, object_spilling_dir)
    if autoscale:
        from ray.autoscaler._private.spark.spark_job_server import _start_spark_job_server
        spark_job_server = _start_spark_job_server(ray_head_ip, spark_job_server_port, spark)
        autoscaling_cluster = AutoscalingCluster(head_resources={'CPU': num_cpus_head_node, 'GPU': num_gpus_head_node, 'memory': heap_memory_head_node, 'object_store_memory': object_store_memory_head_node}, worker_node_types={'ray.worker': {'resources': {'CPU': num_cpus_worker_node, 'GPU': num_gpus_worker_node, 'memory': heap_memory_worker_node, 'object_store_memory': object_store_memory_worker_node}, 'node_config': {}, 'min_workers': 0, 'max_workers': num_worker_nodes}}, extra_provider_config={'ray_head_ip': ray_head_ip, 'ray_head_port': ray_head_port, 'cluster_unique_id': cluster_unique_id, 'using_stage_scheduling': using_stage_scheduling, 'ray_temp_dir': ray_temp_dir, 'worker_node_options': worker_node_options, 'collect_log_to_path': collect_log_to_path, 'spark_job_server_port': spark_job_server_port}, upscaling_speed=autoscale_upscaling_speed, idle_timeout_minutes=autoscale_idle_timeout_minutes)
        ray_head_proc, tail_output_deque = autoscaling_cluster.start(ray_head_ip, ray_head_port, ray_client_server_port, ray_temp_dir, dashboard_options, head_node_options, collect_log_to_path)
        ray_head_node_cmd = autoscaling_cluster.ray_head_node_cmd
    else:
        worker_port_range_begin, worker_port_range_end = _preallocate_ray_worker_port_range()
        ray_head_node_cmd = [sys.executable, '-m', 'ray.util.spark.start_ray_node', f'--temp-dir={ray_temp_dir}', '--block', '--head', f'--node-ip-address={ray_head_ip}', f'--port={ray_head_port}', f'--ray-client-server-port={ray_client_server_port}', f'--num-cpus={num_cpus_head_node}', f'--num-gpus={num_gpus_head_node}', f'--memory={heap_memory_head_node}', f'--object-store-memory={object_store_memory_head_node}', f'--min-worker-port={worker_port_range_begin}', f'--max-worker-port={worker_port_range_end - 1}', *dashboard_options, *_convert_ray_node_options(head_node_options)]
        _logger.info(f'Starting Ray head, command: {' '.join(ray_head_node_cmd)}')
        ray_head_proc, tail_output_deque = exec_cmd(ray_head_node_cmd, synchronous=False, extra_env={RAY_ON_SPARK_COLLECT_LOG_TO_PATH: collect_log_to_path or '', RAY_ON_SPARK_START_RAY_PARENT_PID: str(os.getpid())})
        spark_job_server = None
    time.sleep(_RAY_HEAD_STARTUP_TIMEOUT)
    if not is_port_in_use(ray_head_ip, ray_head_port):
        if ray_head_proc.poll() is None:
            ray_head_proc.terminate()
            time.sleep(0.5)
        cmd_exec_failure_msg = gen_cmd_exec_failure_msg(ray_head_node_cmd, ray_head_proc.returncode, tail_output_deque)
        raise RuntimeError('Start Ray head node failed!\n' + cmd_exec_failure_msg)
    _logger.info('Ray head node started.')
    cluster_address = f'{ray_head_ip}:{ray_head_port}'
    os.environ['RAY_ADDRESS'] = cluster_address
    ray_cluster_handler = RayClusterOnSpark(autoscale=autoscale, address=cluster_address, head_proc=ray_head_proc, spark_job_group_id=None, num_workers_node=num_worker_nodes, temp_dir=ray_temp_dir, cluster_unique_id=cluster_unique_id, start_hook=start_hook, ray_dashboard_port=ray_dashboard_port, spark_job_server=spark_job_server, ray_client_server_port=ray_client_server_port)
    if not autoscale:
        spark_job_group_id = f'ray-cluster-{ray_head_port}-{cluster_unique_id}'
        ray_cluster_handler.spark_job_group_id = spark_job_group_id

        def background_job_thread_fn():
            try:
                _start_ray_worker_nodes(spark=spark, spark_job_group_id=spark_job_group_id, spark_job_group_desc=f'This job group is for spark job which runs the Ray cluster with ray head node {ray_head_ip}:{ray_head_port}', num_worker_nodes=num_worker_nodes, using_stage_scheduling=using_stage_scheduling, ray_head_ip=ray_head_ip, ray_head_port=ray_head_port, ray_temp_dir=ray_temp_dir, num_cpus_per_node=num_cpus_worker_node, num_gpus_per_node=num_gpus_worker_node, heap_memory_per_node=heap_memory_worker_node, object_store_memory_per_node=object_store_memory_worker_node, worker_node_options=worker_node_options, collect_log_to_path=collect_log_to_path, autoscale_mode=False, spark_job_server_port=spark_job_server_port)
            except Exception as e:
                if not ray_cluster_handler.spark_job_is_canceled:
                    ray_cluster_handler.background_job_exception = e
                    ray_cluster_handler.shutdown(cancel_background_job=False)
        try:
            threading.Thread(target=inheritable_thread_target(background_job_thread_fn), args=()).start()
            start_hook.on_cluster_created(ray_cluster_handler)
            for _ in range(_BACKGROUND_JOB_STARTUP_WAIT):
                time.sleep(1)
                if ray_cluster_handler.background_job_exception is not None:
                    raise RuntimeError('Ray workers failed to start.') from ray_cluster_handler.background_job_exception
        except Exception:
            ray_cluster_handler.shutdown()
            raise
    return ray_cluster_handler