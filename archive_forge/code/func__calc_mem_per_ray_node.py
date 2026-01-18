import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _calc_mem_per_ray_node(available_physical_mem_per_node, available_shared_mem_per_node, configured_object_store_bytes):
    from ray._private.ray_constants import DEFAULT_OBJECT_STORE_MEMORY_PROPORTION, OBJECT_STORE_MINIMUM_MEMORY_BYTES
    warning_msg = None
    object_store_bytes = configured_object_store_bytes or available_physical_mem_per_node * DEFAULT_OBJECT_STORE_MEMORY_PROPORTION
    if object_store_bytes > available_shared_mem_per_node:
        object_store_bytes = available_shared_mem_per_node
    object_store_bytes_upper_bound = available_physical_mem_per_node * _RAY_ON_SPARK_MAX_OBJECT_STORE_MEMORY_PROPORTION
    if object_store_bytes > object_store_bytes_upper_bound:
        object_store_bytes = object_store_bytes_upper_bound
        warning_msg = 'Your configured `object_store_memory_per_node` value is too high and it is capped by 80% of per-Ray node allocated memory.'
    if object_store_bytes < OBJECT_STORE_MINIMUM_MEMORY_BYTES:
        object_store_bytes = OBJECT_STORE_MINIMUM_MEMORY_BYTES
        warning_msg = f'Your operating system is configured with too small /dev/shm size, so `object_store_memory_per_node` value is configured to minimal size ({OBJECT_STORE_MINIMUM_MEMORY_BYTES} bytes),Please increase system /dev/shm size.'
    object_store_bytes = int(object_store_bytes)
    heap_mem_bytes = available_physical_mem_per_node - object_store_bytes
    return (heap_mem_bytes, object_store_bytes, warning_msg)