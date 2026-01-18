import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
def _update_cloudwatch_config(self, config_type: str, is_head_node: bool) -> None:
    """
        check whether update operations are needed in
        cloudwatch related configs
        """
    cwa_installed = self._setup_cwa()
    param_name = self._get_ssm_param_name(config_type)
    if cwa_installed:
        if is_head_node:
            cw_config_ssm = self._set_cloudwatch_ssm_config_param(param_name, config_type)
            cur_cw_config_hash = self._sha1_hash_file(config_type)
            ssm_cw_config_hash = self._sha1_hash_json(cw_config_ssm)
            if cur_cw_config_hash != ssm_cw_config_hash:
                logger.info('Cloudwatch {} config file has changed.'.format(config_type))
                self._upload_config_to_ssm_and_set_hash_tag(config_type)
                self.CLOUDWATCH_CONFIG_TYPE_TO_UPDATE_FUNC_HEAD_NODE.get(config_type)()
        else:
            head_node_hash = self._get_head_node_config_hash(config_type)
            cur_node_hash = self._get_cur_node_config_hash(config_type)
            if head_node_hash != cur_node_hash:
                logger.info('Cloudwatch {} config file has changed.'.format(config_type))
                update_func = self.CLOUDWATCH_CONFIG_TYPE_TO_UPDATE_FUNC_WORKER_NODE.get(config_type)
                if update_func:
                    update_func()
                self._update_cloudwatch_hash_tag_value(self.node_id, head_node_hash, config_type)