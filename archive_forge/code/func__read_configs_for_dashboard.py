import copy
from dataclasses import asdict
import json
import os
from typing import List, Tuple
import ray
from ray.dashboard.modules.metrics.dashboards.common import DashboardConfig, Panel
from ray.dashboard.modules.metrics.dashboards.default_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.serve_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.serve_deployment_dashboard_panels import (
from ray.dashboard.modules.metrics.dashboards.data_dashboard_panels import (
def _read_configs_for_dashboard(dashboard_config: DashboardConfig) -> Tuple[str, List[str]]:
    """
    Reads environment variable configs for overriding uid or global_filters for a given
    dashboard.

    Returns:
      Tuple with format uid, global_filters
    """
    uid = os.environ.get(GRAFANA_DASHBOARD_UID_OVERRIDE_ENV_VAR_TEMPLATE.format(name=dashboard_config.name)) or dashboard_config.default_uid
    global_filters_str = os.environ.get(GRAFANA_DASHBOARD_GLOBAL_FILTERS_OVERRIDE_ENV_VAR_TEMPLATE.format(name=dashboard_config.name)) or ''
    global_filters = global_filters_str.split(',')
    return (uid, global_filters)