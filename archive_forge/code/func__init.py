import time
from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib
def _init():
    """Initialize the metrics mapping."""
    global _METRICS_MAPPING
    execution_time_buckets = monitoring.ExponentialBuckets(scale=0.1, growth_factor=10, bucket_count=6)
    tracing_time_buckets = execution_time_buckets
    fetch_time_buckets = monitoring.ExponentialBuckets(scale=0.001, growth_factor=10, bucket_count=7)
    server_update_time_buckets = monitoring.ExponentialBuckets(scale=1, growth_factor=10, bucket_count=5)
    function_tracing_sampler = monitoring.Sampler('/tensorflow/api/ps_strategy/coordinator/function_tracing', tracing_time_buckets, 'Sampler to track the time (in seconds) for tracing functions.')
    closure_execution_sampler = monitoring.Sampler('/tensorflow/api/ps_strategy/coordinator/closure_execution', execution_time_buckets, 'Sampler to track the time (in seconds) for executing closures.')
    remote_value_fetch_sampler = monitoring.Sampler('/tensorflow/api/ps_strategy/coordinator/remote_value_fetch', fetch_time_buckets, 'Sampler to track the time (in seconds) for fetching remote_value.')
    server_def_update_sampler = monitoring.Sampler('/tensorflow/api/ps_strategy/coordinator/server_def_update', server_update_time_buckets, 'Sample to track the time (in seconds) for updating the server def upon worker recovery.')
    queued_closure_gauge = monitoring.IntGauge('/tensorflow/api/ps_strategy/coordinator/queued_closures', 'Track how many closures are in the coordinator queue pending execution.')
    inflight_closure_gauge = monitoring.IntGauge('/tensorflow/api/ps_strategy/coordinator/inflight_closures', 'Track how many closures are currently being processed by workers.')
    worker_failure_counter = monitoring.Counter('/tensorflow/api/ps_strategy/coordinator/recoverable_worker_failure_count', 'Track how many recoverable worker failures have been encountered.')
    _METRICS_MAPPING = {'function_tracing': function_tracing_sampler, 'closure_execution': closure_execution_sampler, 'remote_value_fetch': remote_value_fetch_sampler, 'server_def_update': server_def_update_sampler, 'queued_closures': queued_closure_gauge, 'inflight_closures': inflight_closure_gauge, 'worker_failures': worker_failure_counter}