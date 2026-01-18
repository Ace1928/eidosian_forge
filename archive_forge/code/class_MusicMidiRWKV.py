from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
class MusicMidiRWKV(AbstractRWKV):

    def __init__(self, model, pipeline):
        super().__init__(model, pipeline)
        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.8
        self.top_k = 8
        self.rwkv_type = RWKVType.Music

    def adjust_occurrence(self, occurrence: Dict, token: int):
        for n in occurrence:
            occurrence[n] *= 0.997
        if token >= 128 or token == 127:
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
        else:
            occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        for n in occurrence:
            logits[n] -= 0 + occurrence[n] * 0.5
        logits[0] += (i - 2000) / 500
        logits[127] -= 1

    def fix_tokens(self, tokens) -> List[int]:
        return tokens

    def run_rnn(self, _tokens: List[str], newline_adj: int=0) -> Tuple[List[float], int]:
        tokens = [int(x) for x in _tokens]
        token_len = len(tokens)
        self.model_tokens += tokens
        out, self.model_state = self.model.forward(tokens, self.model_state)
        return (out, token_len)

    def delta_postprocess(self, delta: str) -> str:
        return ' ' + delta