from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
def _reduce_chat(all_chunks: List[ChatCompletionChunk]) -> dict:
    choices_by_index: DefaultDict[int, List[Choice]] = defaultdict(list)
    for chunk in all_chunks:
        for choice in chunk.choices:
            choices_by_index[choice.index].append(choice)
    if all_chunks:
        d = all_chunks[-1].model_dump()
        d['choices'] = [_reduce_choices(choices) for choices in choices_by_index.values()]
    else:
        d = {'choices': [{'message': {'role': 'assistant', 'content': ''}}]}
    return d