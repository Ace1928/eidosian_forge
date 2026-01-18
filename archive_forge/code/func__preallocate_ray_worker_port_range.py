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
def _preallocate_ray_worker_port_range():
    """
    If we start multiple ray workers on a machine concurrently, some ray worker
    processes might fail due to ray port conflicts, this is because race condition
    on getting free port and opening the free port.
    To address the issue, this function use an exclusive file lock to delay the
    worker processes to ensure that port acquisition does not create a resource
    contention issue due to a race condition.

    After acquiring lock, it will allocate port range for worker ports
    (for ray node config --min-worker-port and --max-worker-port).
    Because on a spark cluster, multiple ray cluster might be created, so on one spark
    worker machine, there might be multiple ray worker nodes running, these worker
    nodes might belong to different ray cluster, and we must ensure these ray nodes on
    the same machine using non-overlapping worker port range, to achieve this, in this
    function, it creates a file `/tmp/ray_on_spark_worker_port_allocation.txt` file,
    the file format is composed of multiple lines, each line contains 2 number: `pid`
    and `port_range_slot_index`, each port range slot allocates 1000 ports, and
    corresponding port range is:
     - range_begin (inclusive): 20000 + port_range_slot_index * 1000
     - range_end (exclusive): range_begin + 1000
    In this function, it first scans `/tmp/ray_on_spark_worker_port_allocation.txt`
    file, removing lines that containing dead process pid, then find the first unused
    port_range_slot_index, then regenerate this file, and return the allocated port
    range.

    Returns: Allocated port range for current worker ports
    """
    import psutil
    import fcntl

    def acquire_lock(file_path):
        mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        try:
            fd = os.open(file_path, mode)
            os.chmod(file_path, 511)
            max_lock_iter = 600
            for _ in range(max_lock_iter):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    return fd
                time.sleep(10)
            raise TimeoutError(f'Acquiring lock on file {file_path} timeout.')
        except Exception:
            os.close(fd)
    lock_file_path = '/tmp/ray_on_spark_worker_startup_barrier_lock.lock'
    try:
        lock_fd = acquire_lock(lock_file_path)
    except TimeoutError:
        try:
            os.remove(lock_file_path)
        except Exception:
            pass
        lock_fd = acquire_lock(lock_file_path)

    def release_lock():
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    try:
        port_alloc_file = '/tmp/ray_on_spark_worker_port_allocation.txt'
        if os.path.exists(port_alloc_file):
            with open(port_alloc_file, mode='r') as fp:
                port_alloc_data = fp.read()
            port_alloc_table = [line.split(' ') for line in port_alloc_data.strip().split('\n')]
            port_alloc_table = [(int(pid_str), int(slot_index_str)) for pid_str, slot_index_str in port_alloc_table]
        else:
            port_alloc_table = []
            with open(port_alloc_file, mode='w'):
                pass
            os.chmod(port_alloc_file, 511)
        port_alloc_map = {pid: slot_index for pid, slot_index in port_alloc_table if psutil.pid_exists(pid)}
        allocated_slot_set = set(port_alloc_map.values())
        if len(allocated_slot_set) == 0:
            new_slot_index = 0
        else:
            new_slot_index = max(allocated_slot_set) + 1
            for index in range(new_slot_index):
                if index not in allocated_slot_set:
                    new_slot_index = index
                    break
        port_alloc_map[os.getpid()] = new_slot_index
        with open(port_alloc_file, mode='w') as fp:
            for pid, slot_index in port_alloc_map.items():
                fp.write(f'{pid} {slot_index}\n')
        worker_port_range_begin = 20000 + new_slot_index * 1000
        worker_port_range_end = worker_port_range_begin + 1000
        if worker_port_range_end > 65536:
            raise RuntimeError('Too many ray worker nodes are running on this machine, cannot allocate worker port range for new ray worker node.')
    except Exception:
        release_lock()
        raise

    def hold_lock():
        time.sleep(_RAY_WORKER_NODE_STARTUP_INTERVAL)
        release_lock()
    threading.Thread(target=hold_lock, args=()).start()
    return (worker_port_range_begin, worker_port_range_end)