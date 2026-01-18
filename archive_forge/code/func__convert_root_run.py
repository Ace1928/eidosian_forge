import collections
import concurrent.futures
import datetime
import itertools
import uuid
from typing import DefaultDict, List, Optional, Sequence, Tuple, TypeVar
import langsmith.beta._utils as beta_utils
import langsmith.schemas as ls_schemas
from langsmith import evaluation as ls_eval
from langsmith.client import Client
def _convert_root_run(root: ls_schemas.Run, run_to_example_map: dict) -> List[dict]:
    """Convert the root run and its child runs to a list of dictionaries.

    Parameters:
    - root (ls_schemas.Run): The root run to convert.
    - run_to_example_map (dict): The dictionary mapping run IDs to example IDs.

    Returns:
    - List[dict]: The list of converted run dictionaries.
    """
    runs_ = [root]
    trace_id = uuid.uuid4()
    id_map = {root.trace_id: trace_id}
    results = []
    while runs_:
        src = runs_.pop()
        src_dict = src.dict(exclude={'parent_run_ids', 'child_run_ids', 'session_id'})
        id_map[src_dict['id']] = id_map.get(src_dict['id'], uuid.uuid4())
        src_dict['id'] = id_map[src_dict['id']]
        src_dict['trace_id'] = id_map[src_dict['trace_id']]
        if src.child_runs:
            runs_.extend(src.child_runs)
        results.append(src_dict)
    result = [_convert_ids(r, id_map) for r in results]
    result[0]['reference_example_id'] = run_to_example_map[root.id]
    return result