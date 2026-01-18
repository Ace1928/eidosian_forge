from typing import List
import numpy as np
def _split_gen_kwargs(gen_kwargs: dict, max_num_jobs: int) -> List[dict]:
    """Split the gen_kwargs into `max_num_job` gen_kwargs"""
    num_shards = _number_of_shards_in_gen_kwargs(gen_kwargs)
    if num_shards == 1:
        return [dict(gen_kwargs)]
    else:
        shard_indices_per_group = _distribute_shards(num_shards=num_shards, max_num_jobs=max_num_jobs)
        return [{key: [value[shard_idx] for shard_idx in shard_indices_per_group[group_idx]] if isinstance(value, list) else value for key, value in gen_kwargs.items()} for group_idx in range(len(shard_indices_per_group))]