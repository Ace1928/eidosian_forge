import decimal
import json
import logging
import time
from typing import Any, Dict, Optional
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.kuberay import node_provider, utils
from ray.autoscaler._private.util import validate_config
class AutoscalingConfigProducer:
    """Produces an autoscaling config by reading data from the RayCluster CR.

    Used to fetch the autoscaling config at the beginning of each autoscaler iteration.

    In the context of Ray deployment on Kubernetes, the autoscaling config is an
    internal interface.

    The autoscaling config carries the strict subset of RayCluster CR data required by
    the autoscaler to make scaling decisions; in particular, the autoscaling config does
    not carry pod configuration data.

    This class is the only public object in this file.
    """

    def __init__(self, ray_cluster_name, ray_cluster_namespace):
        self._headers, self._verify = node_provider.load_k8s_secrets()
        self._ray_cr_url = node_provider.url_from_resource(namespace=ray_cluster_namespace, path=f'rayclusters/{ray_cluster_name}')

    def __call__(self):
        ray_cr = self._fetch_ray_cr_from_k8s_with_retries()
        autoscaling_config = _derive_autoscaling_config_from_ray_cr(ray_cr)
        return autoscaling_config

    def _fetch_ray_cr_from_k8s_with_retries(self) -> Dict[str, Any]:
        """Fetch the RayCluster CR by querying the K8s API server.

        Retry on HTTPError for robustness, in particular to protect autoscaler
        initialization.
        """
        for i in range(1, MAX_RAYCLUSTER_FETCH_TRIES + 1):
            try:
                return self._fetch_ray_cr_from_k8s()
            except requests.HTTPError as e:
                if i < MAX_RAYCLUSTER_FETCH_TRIES:
                    logger.exception('Failed to fetch RayCluster CR from K8s. Retrying.')
                    time.sleep(RAYCLUSTER_FETCH_RETRY_S)
                else:
                    raise e from None
        raise AssertionError

    def _fetch_ray_cr_from_k8s(self) -> Dict[str, Any]:
        result = requests.get(self._ray_cr_url, headers=self._headers, verify=self._verify)
        if not result.status_code == 200:
            result.raise_for_status()
        ray_cr = result.json()
        return ray_cr