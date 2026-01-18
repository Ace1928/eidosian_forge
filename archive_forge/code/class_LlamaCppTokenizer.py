import math
from typing import TYPE_CHECKING, Dict, Optional, Set, Type, Union
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from outlines.fsm.guide import CFGGuide, Guide, RegexGuide
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.integrations.utils import convert_json_schema_to_str
class LlamaCppTokenizer:

    def __init__(self, model: 'Llama'):
        self.eos_token_id = model.token_eos()
        self.pad_token_id = self.eos_token_id
        self.special_tokens: Set[int] = set()
        self.vocabulary: Dict[str, int] = dict()
        for t in range(model.n_vocab()):
            token_piece = model.tokenizer().decode([t])
            self.vocabulary[token_piece] = t

    def convert_token_to_string(self, token: str) -> str:
        return token