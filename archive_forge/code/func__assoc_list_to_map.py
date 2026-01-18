import concurrent.futures
from threading import RLock
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
def _assoc_list_to_map(lst):
    """
    Convert an association list to a dictionary.
    """
    d = {}
    for run_id, metric in lst:
        d[run_id] = d[run_id] + [metric] if run_id in d else [metric]
    return d