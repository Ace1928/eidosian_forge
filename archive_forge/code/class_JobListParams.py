from __future__ import annotations
from typing_extensions import TypedDict
class JobListParams(TypedDict, total=False):
    after: str
    'Identifier for the last job from the previous pagination request.'
    limit: int
    'Number of fine-tuning jobs to retrieve.'