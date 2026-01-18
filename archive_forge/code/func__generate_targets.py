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
def _generate_targets(panel: Panel, panel_global_filters: List[str]) -> List[dict]:
    targets = []
    for target, ref_id in zip(panel.targets, gen_incrementing_alphabets(len(panel.targets))):
        template = copy.deepcopy(TARGET_TEMPLATE)
        template.update({'expr': target.expr.format(global_filters=','.join(panel_global_filters)), 'legendFormat': target.legend, 'refId': ref_id})
        targets.append(template)
    return targets