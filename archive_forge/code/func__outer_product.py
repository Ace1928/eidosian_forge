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
def _outer_product(list1: List[T], list2: List[U]) -> List[Tuple[T, U]]:
    return list(itertools.product(list1, list2))