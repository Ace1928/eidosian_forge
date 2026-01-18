import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _calc_mem_per_ray_worker_node(num_task_slots, physical_mem_bytes, shared_mem_bytes, configured_object_store_bytes):
    available_physical_mem_per_node = int(physical_mem_bytes / num_task_slots * _RAY_ON_SPARK_NODE_MEMORY_BUFFER_OFFSET)
    available_shared_mem_per_node = int(shared_mem_bytes / num_task_slots * _RAY_ON_SPARK_NODE_MEMORY_BUFFER_OFFSET)
    return _calc_mem_per_ray_node(available_physical_mem_per_node, available_shared_mem_per_node, configured_object_store_bytes)