import os, sys, types, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from lm_eval import tasks, evaluator
from lm_eval.models.gpt2 import GPT2LM
class TokenizerWrapper:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)