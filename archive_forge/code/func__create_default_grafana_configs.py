import asyncio
import aiohttp
import logging
import os
import shutil
from typing import Optional
import psutil
from urllib.parse import quote
from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
from ray.dashboard.modules.metrics.grafana_datasource_template import (
from ray.dashboard.modules.metrics.grafana_dashboard_provisioning_template import (
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.consts import AVAILABLE_COMPONENT_NAMES_FOR_METRICS
def _create_default_grafana_configs(self):
    """
        Creates the grafana configurations that are by default provided by Ray.
        """
    grafana_config_output_path = os.path.join(self._metrics_root, 'grafana')
    if os.path.exists(grafana_config_output_path):
        shutil.rmtree(grafana_config_output_path)
    os.makedirs(os.path.dirname(grafana_config_output_path), exist_ok=True)
    shutil.copytree(GRAFANA_CONFIG_INPUT_PATH, grafana_config_output_path)
    dashboard_provisioning_path = os.path.join(grafana_config_output_path, 'provisioning', 'dashboards')
    os.makedirs(dashboard_provisioning_path, exist_ok=True)
    with open(os.path.join(dashboard_provisioning_path, 'default.yml'), 'w') as f:
        f.write(DASHBOARD_PROVISIONING_TEMPLATE.format(dashboard_output_folder=self._grafana_dashboard_output_dir))
    prometheus_host = os.environ.get(PROMETHEUS_HOST_ENV_VAR, DEFAULT_PROMETHEUS_HOST)
    data_sources_path = os.path.join(grafana_config_output_path, 'provisioning', 'datasources')
    os.makedirs(data_sources_path, exist_ok=True)
    os.makedirs(self._grafana_dashboard_output_dir, exist_ok=True)
    with open(os.path.join(data_sources_path, 'default.yml'), 'w') as f:
        f.write(GRAFANA_DATASOURCE_TEMPLATE.format(prometheus_host=prometheus_host, prometheus_name=self._prometheus_name))
    with open(os.path.join(self._grafana_dashboard_output_dir, 'default_grafana_dashboard.json'), 'w') as f:
        content, self._dashboard_uids['default'] = generate_default_grafana_dashboard()
        f.write(content)
    with open(os.path.join(self._grafana_dashboard_output_dir, 'serve_grafana_dashboard.json'), 'w') as f:
        content, self._dashboard_uids['serve'] = generate_serve_grafana_dashboard()
        f.write(content)
    with open(os.path.join(self._grafana_dashboard_output_dir, 'serve_deployment_grafana_dashboard.json'), 'w') as f:
        content, self._dashboard_uids['serve_deployment'] = generate_serve_deployment_grafana_dashboard()
        f.write(content)
    with open(os.path.join(self._grafana_dashboard_output_dir, 'data_grafana_dashboard.json'), 'w') as f:
        content, self._dashboard_uids['data'] = generate_data_grafana_dashboard()
        f.write(content)