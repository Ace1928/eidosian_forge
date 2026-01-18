import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
@staticmethod
def _convert_metrics_to_dict(metrics: Metrics) -> Dict[str, Any]:
    """Converts metrics to a dict."""
    metrics_dict = {'stats': metrics.stats, 'trace': metrics.trace}
    usage_stats = {f'usage/{k}': v for k, v in asdict(metrics.usage).items()}
    metrics_dict.update(usage_stats)
    return metrics_dict