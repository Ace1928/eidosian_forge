import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager
def _assign_row_partitions_to_actors(actors: List, row_partitions, data_for_aligning=None):
    """
    Assign row_partitions to actors.

    `row_partitions` will be assigned to actors according to their IPs.
    If distribution isn't even, partitions will be moved from actor
    with excess partitions to actor with lack of them.

    Parameters
    ----------
    actors : list
        List of used actors.
    row_partitions : list
        Row partitions of data to assign.
    data_for_aligning : dict, optional
        Data according to the order of which should be
        distributed `row_partitions`. Used to align y with X.

    Returns
    -------
    dict
        Dictionary of assigned to actors partitions
        as {actor_rank: (partitions, order)}.
    """
    num_actors = len(actors)
    if data_for_aligning is None:
        parts_ips_ref, parts_ref = zip(*row_partitions)
        actor_ips = defaultdict(list)
        for rank, (ip, _) in enumerate(actors):
            actor_ips[ip].append(rank)
        init_parts_distribution = defaultdict(list)
        for idx, (ip, part_ref) in enumerate(zip(RayWrapper.materialize(list(parts_ips_ref)), parts_ref)):
            init_parts_distribution[ip].append((part_ref, idx))
        num_parts = len(parts_ref)
        min_parts_per_actor = math.floor(num_parts / num_actors)
        max_parts_per_actor = math.ceil(num_parts / num_actors)
        num_actors_with_max_parts = num_parts % num_actors
        row_partitions_by_actors = defaultdict(list)
        for actor_ip, ranks in actor_ips.items():
            for rank in ranks:
                num_parts_on_ip = len(init_parts_distribution[actor_ip])
                if num_parts_on_ip == 0:
                    break
                if num_parts_on_ip >= min_parts_per_actor:
                    if num_parts_on_ip >= max_parts_per_actor and num_actors_with_max_parts > 0:
                        pop_slice = slice(0, max_parts_per_actor)
                        num_actors_with_max_parts -= 1
                    else:
                        pop_slice = slice(0, min_parts_per_actor)
                    row_partitions_by_actors[rank].extend(init_parts_distribution[actor_ip][pop_slice])
                    del init_parts_distribution[actor_ip][pop_slice]
                else:
                    row_partitions_by_actors[rank].extend(init_parts_distribution[actor_ip])
                    init_parts_distribution[actor_ip] = []
        for ip in list(init_parts_distribution):
            if len(init_parts_distribution[ip]) == 0:
                init_parts_distribution.pop(ip)
        init_parts_distribution = [pair for pairs in init_parts_distribution.values() for pair in pairs]
        for rank in range(len(actors)):
            num_parts_on_rank = len(row_partitions_by_actors[rank])
            if num_parts_on_rank == max_parts_per_actor or (num_parts_on_rank == min_parts_per_actor and num_actors_with_max_parts == 0):
                continue
            if num_actors_with_max_parts > 0:
                pop_slice = slice(0, max_parts_per_actor - num_parts_on_rank)
                num_actors_with_max_parts -= 1
            else:
                pop_slice = slice(0, min_parts_per_actor - num_parts_on_rank)
            row_partitions_by_actors[rank].extend(init_parts_distribution[pop_slice])
            del init_parts_distribution[pop_slice]
        if len(init_parts_distribution) != 0:
            raise RuntimeError(f'Not all partitions were ditributed between actors: {len(init_parts_distribution)} left.')
        row_parts_by_ranks = dict()
        for rank, pairs_part_pos in dict(row_partitions_by_actors).items():
            parts, order = zip(*pairs_part_pos)
            row_parts_by_ranks[rank] = (list(parts), list(order))
    else:
        row_parts_by_ranks = {rank: ([], []) for rank in range(len(actors))}
        for rank, (_, order_of_indexes) in data_for_aligning.items():
            row_parts_by_ranks[rank][1].extend(order_of_indexes)
            for row_idx in order_of_indexes:
                row_parts_by_ranks[rank][0].append(row_partitions[row_idx])
    return row_parts_by_ranks