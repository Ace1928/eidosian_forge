import logging
from typing import Optional
import ray
from ray.rllib.utils.annotations import DeveloperAPI
Aggregates filters from remote workers (if use_remote_data_for_update=True).

        Local copy is updated and then broadcasted to all remote evaluators
        (if `update_remote` is True).

        Args:
            local_filters: Filters to be synchronized.
            remotes: Remote evaluators with filters.
            update_remote: Whether to push updates from the local filters to the remote
                workers' filters.
            timeout_seconds: How long to wait for filter to get or set filters
            use_remote_data_for_update: Whether to use the `worker_set`'s remote workers
                to update the local filters. If False, stats from the remote workers
                will not be used and discarded.
        