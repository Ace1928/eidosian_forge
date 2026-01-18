import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _get_avail_mem_per_ray_worker_node(num_cpus_per_node, num_gpus_per_node, object_store_memory_per_node):
    num_cpus = _get_cpu_cores()
    num_task_slots = num_cpus // num_cpus_per_node
    if num_gpus_per_node > 0:
        num_gpus = _get_num_physical_gpus()
        if num_task_slots > num_gpus // num_gpus_per_node:
            num_task_slots = num_gpus // num_gpus_per_node
    physical_mem_bytes = _get_spark_worker_total_physical_memory()
    shared_mem_bytes = _get_spark_worker_total_shared_memory()
    ray_worker_node_heap_mem_bytes, ray_worker_node_object_store_bytes, warning_msg = _calc_mem_per_ray_worker_node(num_task_slots, physical_mem_bytes, shared_mem_bytes, object_store_memory_per_node)
    return (ray_worker_node_heap_mem_bytes, ray_worker_node_object_store_bytes, None, warning_msg)