from __future__ import annotations
from typing_extensions import TypedDict
class CheckpointListParams(TypedDict, total=False):
    after: str
    'Identifier for the last checkpoint ID from the previous pagination request.'
    limit: int
    'Number of checkpoints to retrieve.'