from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional
from ..pipelines import Pipeline, get_supported_tasks, pipeline
from ..utils import logging
from . import BaseTransformersCLICommand
class ServeTokenizeResult(BaseModel):
    """
    Tokenize result model
    """
    tokens: List[str]
    tokens_ids: Optional[List[int]]