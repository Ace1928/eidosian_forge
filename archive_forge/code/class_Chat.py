import enum
import warnings
from typing import Dict
from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import Pipeline, build_pipeline_init_args
class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: Dict):
        for message in messages:
            if not ('role' in message and 'content' in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        self.messages = messages