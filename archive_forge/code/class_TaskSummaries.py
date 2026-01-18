import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
@dataclass
class TaskSummaries:
    summary: Union[Dict[str, TaskSummaryPerFuncOrClassName], List[NestedTaskSummary]]
    total_tasks: int
    total_actor_tasks: int
    total_actor_scheduled: int
    summary_by: str = 'func_name'

    @classmethod
    def to_summary_by_func_name(cls, *, tasks: List[Dict]) -> 'TaskSummaries':
        summary = {}
        total_tasks = 0
        total_actor_tasks = 0
        total_actor_scheduled = 0
        for task in tasks:
            key = task['func_or_class_name']
            if key not in summary:
                summary[key] = TaskSummaryPerFuncOrClassName(func_or_class_name=task['func_or_class_name'], type=task['type'])
            task_summary = summary[key]
            state = task['state']
            if state not in task_summary.state_counts:
                task_summary.state_counts[state] = 0
            task_summary.state_counts[state] += 1
            type_enum = TaskType.DESCRIPTOR.values_by_name[task['type']].number
            if type_enum == TaskType.NORMAL_TASK:
                total_tasks += 1
            elif type_enum == TaskType.ACTOR_CREATION_TASK:
                total_actor_scheduled += 1
            elif type_enum == TaskType.ACTOR_TASK:
                total_actor_tasks += 1
        return TaskSummaries(summary=summary, total_tasks=total_tasks, total_actor_tasks=total_actor_tasks, total_actor_scheduled=total_actor_scheduled, summary_by='func_name')

    @classmethod
    def to_summary_by_lineage(cls, *, tasks: List[Dict], actors: List[Dict]) -> 'TaskSummaries':
        """
        This summarizes tasks by lineage.
        i.e. A task will be grouped with another task if they have the
        same parent.

        This does things in 4 steps.
        Step 1: Iterate through all tasks and keep track of them by id and ownership
        Step 2: Put the tasks in a tree structure based on ownership
        Step 3: Merge together siblings in the tree if there are more
        than one with the same name.
        Step 4: Sort by running and then errored and then successful tasks
        Step 5: Total the children

        This can probably be more efficient if we merge together some steps to
        reduce the amount of iterations but this algorithm produces very easy to
        understand code. We can optimize in the future.
        """
        tasks_by_id = {}
        task_group_by_id = {}
        actor_creation_task_id_for_actor_id = {}
        summary = []
        total_tasks = 0
        total_actor_tasks = 0
        total_actor_scheduled = 0
        for task in tasks:
            tasks_by_id[task['task_id']] = task
            type_enum = TaskType.DESCRIPTOR.values_by_name[task['type']].number
            if type_enum == TaskType.ACTOR_CREATION_TASK:
                actor_creation_task_id_for_actor_id[task['actor_id']] = task['task_id']
        actor_dict = {actor['actor_id']: actor for actor in actors}

        def get_or_create_task_group(task_id: str) -> Optional[NestedTaskSummary]:
            """
            Gets an already created task_group
            OR
            Creates a task group and puts it in the right place under its parent.
            For actor tasks, the parent is the Actor that owns it. For all other
            tasks, the owner is the driver or task that created it.

            Returns None if there is missing data about the task or one of its parents.

            For task groups that represents actors, the id is in the
            format actor:{actor_id}
            """
            if task_id in task_group_by_id:
                return task_group_by_id[task_id]
            task = tasks_by_id.get(task_id)
            if not task:
                logger.debug(f"We're missing data about {task_id}")
                return None
            func_name = task['name'] or task['func_or_class_name']
            task_id = task['task_id']
            type_enum = TaskType.DESCRIPTOR.values_by_name[task['type']].number
            task_group_by_id[task_id] = NestedTaskSummary(name=func_name, key=task_id, type=task['type'], timestamp=task['creation_time_ms'], link=Link(type='task', id=task_id))
            if type_enum == TaskType.ACTOR_TASK or type_enum == TaskType.ACTOR_CREATION_TASK:
                parent_task_group = get_or_create_actor_task_group(task['actor_id'])
                if parent_task_group:
                    parent_task_group.children.append(task_group_by_id[task_id])
            else:
                parent_task_id = task['parent_task_id']
                if not parent_task_id or parent_task_id.startswith(DRIVER_TASK_ID_PREFIX):
                    summary.append(task_group_by_id[task_id])
                else:
                    parent_task_group = get_or_create_task_group(parent_task_id)
                    if parent_task_group:
                        parent_task_group.children.append(task_group_by_id[task_id])
            return task_group_by_id[task_id]

        def get_or_create_actor_task_group(actor_id: str) -> Optional[NestedTaskSummary]:
            """
            Gets an existing task group that represents an actor.
            OR
            Creates a task group that represents an actor. The owner of the actor is
            the parent of the creation_task that created that actor.

            Returns None if there is missing data about the actor or one of its parents.
            """
            key = f'actor:{actor_id}'
            actor = actor_dict.get(actor_id)
            if key not in task_group_by_id:
                creation_task_id = actor_creation_task_id_for_actor_id.get(actor_id)
                creation_task = tasks_by_id.get(creation_task_id)
                if not creation_task:
                    logger.debug(f"We're missing data about actor {actor_id}")
                    return None
                if actor is None:
                    logger.debug(f'We are missing actor info for actor {actor_id}, even though creation task exists: {creation_task}')
                    [actor_name, *rest] = creation_task['func_or_class_name'].split('.')
                else:
                    actor_name = actor['repr_name'] if actor['repr_name'] else actor['class_name']
                task_group_by_id[key] = NestedTaskSummary(name=actor_name, key=key, type='ACTOR', timestamp=task['creation_time_ms'], link=Link(type='actor', id=actor_id))
                parent_task_id = creation_task['parent_task_id']
                if not parent_task_id or parent_task_id.startswith(DRIVER_TASK_ID_PREFIX):
                    summary.append(task_group_by_id[key])
                else:
                    parent_task_group = get_or_create_task_group(parent_task_id)
                    if parent_task_group:
                        parent_task_group.children.append(task_group_by_id[key])
            return task_group_by_id[key]
        for task in tasks:
            task_id = task['task_id']
            task_group = get_or_create_task_group(task_id)
            if not task_group:
                continue
            state = task['state']
            if state not in task_group.state_counts:
                task_group.state_counts[state] = 0
            task_group.state_counts[state] += 1
            type_enum = TaskType.DESCRIPTOR.values_by_name[task['type']].number
            if type_enum == TaskType.NORMAL_TASK:
                total_tasks += 1
            elif type_enum == TaskType.ACTOR_CREATION_TASK:
                total_actor_scheduled += 1
            elif type_enum == TaskType.ACTOR_TASK:
                total_actor_tasks += 1

        def merge_sibings_for_task_group(siblings: List[NestedTaskSummary]) -> Tuple[List[NestedTaskSummary], Optional[int]]:
            """
            Merges task summaries with the same name into a group if there are more than
            one child with that name.

            Args:
                siblings: A list of NestedTaskSummary's to merge together

            Returns
                Index 0: A list of NestedTaskSummary's which have been merged
                Index 1: The smallest timestamp amongst the siblings
            """
            if not len(siblings):
                return (siblings, None)
            groups = {}
            min_timestamp = None
            for child in siblings:
                child.children, child_min_timestamp = merge_sibings_for_task_group(child.children)
                if child_min_timestamp and child_min_timestamp < (child.timestamp or sys.maxsize):
                    child.timestamp = child_min_timestamp
                if child.name not in groups:
                    groups[child.name] = NestedTaskSummary(name=child.name, key=child.name, type='GROUP')
                groups[child.name].children.append(child)
                if child.timestamp and child.timestamp < (groups[child.name].timestamp or sys.maxsize):
                    groups[child.name].timestamp = child.timestamp
                    if child.timestamp < (min_timestamp or sys.maxsize):
                        min_timestamp = child.timestamp
            return ([group if len(group.children) > 1 else group.children[0] for group in groups.values()], min_timestamp)
        summary, _ = merge_sibings_for_task_group(summary)

        def get_running_tasks_count(task_group: NestedTaskSummary) -> int:
            return task_group.state_counts.get('RUNNING', 0) + task_group.state_counts.get('RUNNING_IN_RAY_GET', 0) + task_group.state_counts.get('RUNNING_IN_RAY_WAIT', 0)

        def get_pending_tasks_count(task_group: NestedTaskSummary) -> int:
            return task_group.state_counts.get('PENDING_ARGS_AVAIL', 0) + task_group.state_counts.get('PENDING_NODE_ASSIGNMENT', 0) + task_group.state_counts.get('PENDING_OBJ_STORE_MEM_AVAIL', 0) + task_group.state_counts.get('PENDING_ARGS_FETCH', 0)

        def sort_task_groups(task_groups: List[NestedTaskSummary]) -> None:
            task_groups.sort(key=lambda x: 0 if x.type == 'ACTOR_CREATION_TASK' else 1)
            task_groups.sort(key=lambda x: x.timestamp or sys.maxsize)
            task_groups.sort(key=lambda x: x.state_counts.get('FAIELD', 0), reverse=True)
            task_groups.sort(key=get_pending_tasks_count, reverse=True)
            task_groups.sort(key=get_running_tasks_count, reverse=True)

        def calc_total_for_task_group(task_group: NestedTaskSummary) -> NestedTaskSummary:
            """
            Calculates the total of a group as the sum of all children.
            Sorts children by timestamp
            """
            if not len(task_group.children):
                return task_group
            for child in task_group.children:
                totaled = calc_total_for_task_group(child)
                for state, count in totaled.state_counts.items():
                    task_group.state_counts[state] = task_group.state_counts.get(state, 0) + count
            sort_task_groups(task_group.children)
            return task_group
        summary = [calc_total_for_task_group(task_group) for task_group in summary]
        sort_task_groups(summary)
        return TaskSummaries(summary=summary, total_tasks=total_tasks, total_actor_tasks=total_actor_tasks, total_actor_scheduled=total_actor_scheduled, summary_by='lineage')