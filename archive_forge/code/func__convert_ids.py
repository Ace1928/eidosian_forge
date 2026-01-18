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
def _convert_ids(run_dict: dict, id_map: dict):
    """Convert the IDs in the run dictionary using the provided ID map.

    Parameters:
    - run_dict (dict): The dictionary representing a run.
    - id_map (dict): The dictionary mapping old IDs to new IDs.

    Returns:
    - dict: The updated run dictionary.
    """
    do = run_dict['dotted_order']
    for k, v in id_map.items():
        do = do.replace(str(k), str(v))
    run_dict['dotted_order'] = do
    if run_dict.get('parent_run_id'):
        run_dict['parent_run_id'] = id_map[run_dict['parent_run_id']]
    if not run_dict.get('extra'):
        run_dict['extra'] = {}
    return run_dict