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
class TextRWKV(AbstractRWKV):

    def __init__(self, model, pipeline) -> None:
        super().__init__(model, pipeline)
        self.CHUNK_LEN = 256
        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.3
        self.top_k = 0
        self.penalty_alpha_presence = 0
        self.penalty_alpha_frequency = 1
        self.interface = ':'
        if self.tokenizer_len < 65536:
            self.rwkv_type = RWKVType.Raven
            self.user = 'Bob'
            self.bot = 'Alice'
            self.END_OF_LINE = 187
        else:
            self.rwkv_type = RWKVType.World
            self.user = 'User'
            self.bot = 'Assistant'
            self.END_OF_LINE = 11
        self.AVOID_REPEAT_TOKENS = set()
        AVOID_REPEAT = '，：？！'
        for i in AVOID_REPEAT:
            dd = self.pipeline.encode(i)
            assert len(dd) == 1
            self.AVOID_REPEAT_TOKENS.add(dd[0])
        self.AVOID_PENALTY_TOKENS = set()
        AVOID_PENALTY = '\n,.:?!，。：？！"“”<>[]{}/\\|;；~`@#$%^&*()_+-=0123456789 '
        for i in AVOID_PENALTY:
            dd = self.pipeline.encode(i)
            if len(dd) == 1:
                self.AVOID_PENALTY_TOKENS.add(dd[0])
        self.__preload()

    def adjust_occurrence(self, occurrence: Dict, token: int):
        for xxx in occurrence:
            occurrence[xxx] *= self.penalty_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        for n in occurrence:
            logits[n] -= self.penalty_alpha_presence + occurrence[n] * self.penalty_alpha_frequency
        if self.global_penalty and i == 0:
            for token in self.model_tokens:
                token = int(token)
                if token not in self.AVOID_PENALTY_TOKENS:
                    self.adjust_occurrence(occurrence, token)

    def fix_tokens(self, tokens) -> List[int]:
        if self.rwkv_type == RWKVType.World:
            return tokens
        if len(tokens) > 0 and tokens[-1] == 535:
            tokens = tokens[:-1] + [self.END_OF_LINE, self.END_OF_LINE]
        return tokens

    def run_rnn(self, _tokens: List[str], newline_adj: int=0) -> Tuple[List[float], int]:
        tokens = [int(x) for x in _tokens]
        token_len = len(tokens)
        self.model_tokens += tokens
        while len(tokens) > 0:
            out, self.model_state = self.model.forward(tokens[:self.CHUNK_LEN], self.model_state)
            tokens = tokens[self.CHUNK_LEN:]
        out[self.END_OF_LINE] += newline_adj
        if self.model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
            out[self.model_tokens[-1]] = -999999999
        return (out, token_len)

    def delta_postprocess(self, delta: str) -> str:
        return delta

    def __preload(self):
        interface = self.interface
        user = self.user
        bot = self.bot
        preset_system = f"\nThe following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. {bot} is very intelligent, creative and friendly. {bot} is unlikely to disagree with {user}, and {bot} doesn't like to ask {user} questions. {bot} likes to tell {user} a lot about herself and her opinions. {bot} usually gives {user} kind, helpful and informative advices.\n\n" if self.rwkv_type == RWKVType.Raven else f'{user}{interface} hi\n\n{bot}{interface} Hi. ' + 'I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.\n\n'
        logits, _ = self.run_rnn(self.fix_tokens(self.pipeline.encode(preset_system)))
        try:
            state_cache.add_state(state_cache.AddStateBody(prompt=preset_system, tokens=self.model_tokens, state=self.model_state, logits=logits))
        except HTTPException:
            pass