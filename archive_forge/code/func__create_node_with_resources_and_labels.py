import json
import logging
import sys
from threading import RLock
from typing import Any, Dict, Optional
import requests
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _create_node_with_resources_and_labels(self, node_config, tags, resources, labels):
    from ray.util.spark.cluster_init import _append_resources_config
    with self.lock:
        resources = resources.copy()
        node_type = tags[TAG_RAY_USER_NODE_TYPE]
        node_id = str(self.get_next_node_id())
        resources['NODE_ID_AS_RESOURCE'] = int(node_id)
        conf = self.provider_config.copy()
        num_cpus_per_node = resources.pop('CPU')
        num_gpus_per_node = resources.pop('GPU')
        heap_memory_per_node = resources.pop('memory')
        object_store_memory_per_node = resources.pop('object_store_memory')
        conf['worker_node_options'] = _append_resources_config(conf['worker_node_options'], resources)
        response = requests.post(url=self.spark_job_server_url + '/create_node', json={'spark_job_group_id': self._gen_spark_job_group_id(node_id), 'spark_job_group_desc': f'This job group is for spark job which runs the Ray cluster worker node {node_id} connecting to ray head node {self.ray_head_ip}:{self.ray_head_port}', 'using_stage_scheduling': conf['using_stage_scheduling'], 'ray_head_ip': self.ray_head_ip, 'ray_head_port': self.ray_head_port, 'ray_temp_dir': conf['ray_temp_dir'], 'num_cpus_per_node': num_cpus_per_node, 'num_gpus_per_node': num_gpus_per_node, 'heap_memory_per_node': heap_memory_per_node, 'object_store_memory_per_node': object_store_memory_per_node, 'worker_node_options': conf['worker_node_options'], 'collect_log_to_path': conf['collect_log_to_path']})
        try:
            response.raise_for_status()
        except Exception:
            raise NodeLaunchException('Node creation failure', f'Starting ray worker node {node_id} failed', sys.exc_info())
        self._nodes[node_id] = {'tags': {TAG_RAY_NODE_KIND: NODE_KIND_WORKER, TAG_RAY_USER_NODE_TYPE: node_type, TAG_RAY_NODE_NAME: node_id, TAG_RAY_NODE_STATUS: STATUS_SETTING_UP}}
        logger.info(f'Spark node provider creates node {node_id}.')