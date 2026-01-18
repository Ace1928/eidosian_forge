import datetime
import io
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import wandb
from wandb.sdk.data_types import trace_tree
from wandb.sdk.integration_utils.auto_logging import Response
def _get_metrics_to_log(self, request: Dict[str, Any], response: Response, results: List[Any], time_elapsed: float) -> Metrics:
    model = response.get('model') or request.get('model')
    usage_metrics = self._get_usage_metrics(response, time_elapsed)
    usage = []
    for result in results:
        row = {'request': result.inputs['request'], 'response': result.outputs['response'], 'model': model, 'start_time': datetime.datetime.fromtimestamp(response['created']), 'end_time': datetime.datetime.fromtimestamp(response['created'] + time_elapsed), 'request_id': response.get('id', None), 'api_type': response.get('api_type', 'openai'), 'session_id': wandb.run.id}
        row.update(asdict(usage_metrics))
        usage.append(row)
    usage_table = wandb.Table(columns=list(usage[0].keys()), data=[item.values() for item in usage])
    trace = self.results_to_trace_tree(request, response, results, time_elapsed)
    metrics = Metrics(stats=usage_table, trace=trace, usage=usage_metrics)
    return metrics