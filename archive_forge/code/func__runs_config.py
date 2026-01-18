from typing import Any, Dict, Optional, TypeVar
from ...public import Api as PublicApi
from ...public import PythonMongoishQueryGenerator, QueryGenerator, Runs
from .util import Attr, Base, coalesce, generate_name, nested_get, nested_set
@property
def _runs_config(self) -> dict:
    return {k: v for run in self.runs for k, v in run.config.items()}