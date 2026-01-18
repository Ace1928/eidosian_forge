import asyncio
import datetime
import json
import logging
import os
import socket
import sys
import traceback
import warnings
import psutil
from typing import List, Optional, Tuple
from collections import defaultdict
import ray
import ray._private.services
import ray._private.utils
from ray.dashboard.consts import (
from ray.dashboard.modules.reporter.profile_manager import CpuProfilingManager
import ray.dashboard.modules.reporter.reporter_consts as reporter_consts
import ray.dashboard.utils as dashboard_utils
from opencensus.stats import stats as stats_module
import ray._private.prometheus_exporter as prometheus_exporter
from prometheus_client.core import REGISTRY
from ray._private.metrics_agent import Gauge, MetricsAgent, Record
from ray._private.ray_constants import DEBUG_AUTOSCALING_STATUS
from ray.core.generated import reporter_pb2, reporter_pb2_grpc
from ray.util.debug import log_once
from ray.dashboard import k8s_utils
from ray._raylet import WorkerID
def _record_stats(self, stats, cluster_stats):
    records_reported = []
    ip = stats['ip']
    if 'autoscaler_report' in cluster_stats and self._is_head_node:
        active_nodes = cluster_stats['autoscaler_report']['active_nodes']
        for node_type, active_node_count in active_nodes.items():
            records_reported.append(Record(gauge=METRICS_GAUGES['cluster_active_nodes'], value=active_node_count, tags={'node_type': node_type}))
        failed_nodes = cluster_stats['autoscaler_report']['failed_nodes']
        failed_nodes_dict = {}
        for node_ip, node_type in failed_nodes:
            if node_type in failed_nodes_dict:
                failed_nodes_dict[node_type] += 1
            else:
                failed_nodes_dict[node_type] = 1
        for node_type, failed_node_count in failed_nodes_dict.items():
            records_reported.append(Record(gauge=METRICS_GAUGES['cluster_failed_nodes'], value=failed_node_count, tags={'node_type': node_type}))
        pending_nodes = cluster_stats['autoscaler_report']['pending_nodes']
        pending_nodes_dict = {}
        for node_ip, node_type, status_message in pending_nodes:
            if node_type in pending_nodes_dict:
                pending_nodes_dict[node_type] += 1
            else:
                pending_nodes_dict[node_type] = 1
        for node_type, pending_node_count in pending_nodes_dict.items():
            records_reported.append(Record(gauge=METRICS_GAUGES['cluster_pending_nodes'], value=pending_node_count, tags={'node_type': node_type}))
    cpu_usage = float(stats['cpu'])
    cpu_record = Record(gauge=METRICS_GAUGES['node_cpu_utilization'], value=cpu_usage, tags={'ip': ip})
    cpu_count, _ = stats['cpus']
    cpu_count_record = Record(gauge=METRICS_GAUGES['node_cpu_count'], value=cpu_count, tags={'ip': ip})
    mem_total, mem_available, _, mem_used = stats['mem']
    mem_used_record = Record(gauge=METRICS_GAUGES['node_mem_used'], value=mem_used, tags={'ip': ip})
    mem_available_record = Record(gauge=METRICS_GAUGES['node_mem_available'], value=mem_available, tags={'ip': ip})
    mem_total_record = Record(gauge=METRICS_GAUGES['node_mem_total'], value=mem_total, tags={'ip': ip})
    shm_used = stats['shm']
    if shm_used:
        node_mem_shared = Record(gauge=METRICS_GAUGES['node_mem_shared_bytes'], value=shm_used, tags={'ip': ip})
        records_reported.append(node_mem_shared)
    "\n        {'index': 0,\n        'uuid': 'GPU-36e1567d-37ed-051e-f8ff-df807517b396',\n        'name': 'NVIDIA A10G',\n        'temperature_gpu': 20,\n        'fan_speed': 0,\n        'utilization_gpu': 1,\n        'utilization_enc': 0,\n        'utilization_dec': 0,\n        'power_draw': 51,\n        'enforced_power_limit': 300,\n        'memory_used': 0,\n        'memory_total': 22731,\n        'processes': []}\n        "
    gpus = stats['gpus']
    gpus_available = len(gpus)
    if gpus_available:
        gpu_tags = {'ip': ip}
        for gpu in gpus:
            gpus_utilization, gram_used, gram_total = (0, 0, 0)
            if gpu['utilization_gpu'] is not None:
                gpus_utilization += gpu['utilization_gpu']
            gram_used += gpu['memory_used']
            gram_total += gpu['memory_total']
            gpu_index = gpu.get('index')
            gpu_name = gpu.get('name')
            gram_available = gram_total - gram_used
            if gpu_index is not None:
                gpu_tags = {'ip': ip, 'GpuIndex': str(gpu_index)}
                if gpu_name:
                    gpu_tags['GpuDeviceName'] = gpu_name
                gpus_available_record = Record(gauge=METRICS_GAUGES['node_gpus_available'], value=1, tags=gpu_tags)
                gpus_utilization_record = Record(gauge=METRICS_GAUGES['node_gpus_utilization'], value=gpus_utilization, tags=gpu_tags)
                gram_used_record = Record(gauge=METRICS_GAUGES['node_gram_used'], value=gram_used, tags=gpu_tags)
                gram_available_record = Record(gauge=METRICS_GAUGES['node_gram_available'], value=gram_available, tags=gpu_tags)
                records_reported.extend([gpus_available_record, gpus_utilization_record, gram_used_record, gram_available_record])
    disk_io_stats = stats['disk_io']
    disk_read_record = Record(gauge=METRICS_GAUGES['node_disk_io_read'], value=disk_io_stats[0], tags={'ip': ip})
    disk_write_record = Record(gauge=METRICS_GAUGES['node_disk_io_write'], value=disk_io_stats[1], tags={'ip': ip})
    disk_read_count_record = Record(gauge=METRICS_GAUGES['node_disk_io_read_count'], value=disk_io_stats[2], tags={'ip': ip})
    disk_write_count_record = Record(gauge=METRICS_GAUGES['node_disk_io_write_count'], value=disk_io_stats[3], tags={'ip': ip})
    disk_io_speed_stats = stats['disk_io_speed']
    disk_read_speed_record = Record(gauge=METRICS_GAUGES['node_disk_io_read_speed'], value=disk_io_speed_stats[0], tags={'ip': ip})
    disk_write_speed_record = Record(gauge=METRICS_GAUGES['node_disk_io_write_speed'], value=disk_io_speed_stats[1], tags={'ip': ip})
    disk_read_iops_record = Record(gauge=METRICS_GAUGES['node_disk_read_iops'], value=disk_io_speed_stats[2], tags={'ip': ip})
    disk_write_iops_record = Record(gauge=METRICS_GAUGES['node_disk_write_iops'], value=disk_io_speed_stats[3], tags={'ip': ip})
    used = stats['disk']['/'].used
    free = stats['disk']['/'].free
    disk_utilization = float(used / (used + free)) * 100
    disk_usage_record = Record(gauge=METRICS_GAUGES['node_disk_usage'], value=used, tags={'ip': ip})
    disk_free_record = Record(gauge=METRICS_GAUGES['node_disk_free'], value=free, tags={'ip': ip})
    disk_utilization_percentage_record = Record(gauge=METRICS_GAUGES['node_disk_utilization_percentage'], value=disk_utilization, tags={'ip': ip})
    network_stats = stats['network']
    network_sent_record = Record(gauge=METRICS_GAUGES['node_network_sent'], value=network_stats[0], tags={'ip': ip})
    network_received_record = Record(gauge=METRICS_GAUGES['node_network_received'], value=network_stats[1], tags={'ip': ip})
    network_speed_stats = stats['network_speed']
    network_send_speed_record = Record(gauge=METRICS_GAUGES['node_network_send_speed'], value=network_speed_stats[0], tags={'ip': ip})
    network_receive_speed_record = Record(gauge=METRICS_GAUGES['node_network_receive_speed'], value=network_speed_stats[1], tags={'ip': ip})
    '\n        Record system stats.\n        '
    raylet_stats = stats['raylet']
    if raylet_stats:
        raylet_pid = str(raylet_stats['pid'])
        records_reported.extend(self._generate_system_stats_record([raylet_stats], 'raylet', pid=raylet_pid))
    workers_stats = stats['workers']
    records_reported.extend(self.generate_worker_stats_record(workers_stats))
    agent_stats = stats['agent']
    if agent_stats:
        agent_pid = str(agent_stats['pid'])
        records_reported.extend(self._generate_system_stats_record([agent_stats], 'agent', pid=agent_pid))
    records_reported.extend([cpu_record, cpu_count_record, mem_used_record, mem_available_record, mem_total_record, disk_read_record, disk_write_record, disk_read_count_record, disk_write_count_record, disk_read_speed_record, disk_write_speed_record, disk_read_iops_record, disk_write_iops_record, disk_usage_record, disk_free_record, disk_utilization_percentage_record, network_sent_record, network_received_record, network_send_speed_record, network_receive_speed_record])
    return records_reported