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
class AbstractRWKV(ABC):

    def __init__(self, model, pipeline):
        self.EOS_ID = 0
        self.name = 'rwkv'
        self.version = 4
        self.model = model
        self.pipeline = pipeline
        self.model_state = None
        self.model_tokens = []
        self.rwkv_type: RWKVType = RWKVType.NoneType
        self.tokenizer_len = len(model.w['emb.weight'])
        self.max_tokens_per_generation = 500
        self.temperature = 1
        self.top_p = 0.3
        self.top_k = 0
        self.penalty_alpha_presence = 0
        self.penalty_alpha_frequency = 1
        self.penalty_decay = 0.996
        self.global_penalty = False

    @abstractmethod
    def adjust_occurrence(self, occurrence: Dict, token: int):
        pass

    @abstractmethod
    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int):
        pass

    @abstractmethod
    def fix_tokens(self, tokens) -> List[int]:
        pass

    @abstractmethod
    def run_rnn(self, _tokens: List[str], newline_adj: int=0) -> Tuple[List[float], int]:
        pass

    @abstractmethod
    def delta_postprocess(self, delta: str) -> str:
        pass

    def get_embedding(self, input: str, fast_mode: bool) -> Tuple[List[float], int]:
        import numpy as np
        if fast_mode:
            embedding, token_len = self.__fast_embedding(self.fix_tokens(self.pipeline.encode(input)), None)
        else:
            self.model_state = None
            self.model_tokens = []
            _, token_len = self.run_rnn(self.fix_tokens(self.pipeline.encode(input)))
            embedding = self.model_state[-11].tolist()
        embedding = (embedding / np.linalg.norm(embedding)).tolist()
        return (embedding, token_len)

    def __fast_embedding(self, tokens: List[str], state):
        import torch
        tokens = [int(x) for x in tokens]
        token_len = len(tokens)
        self = self.model
        with torch.no_grad():
            w = self.w
            args = self.args
            if state == None:
                state = [None] * args.n_layer * 5
                for i in range(args.n_layer):
                    dd = self.strategy[i]
                    dev = dd.device
                    atype = dd.atype
                    state[i * 5 + 0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                    state[i * 5 + 1] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                    state[i * 5 + 2] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous()
                    state[i * 5 + 3] = torch.zeros(args.n_embd, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e+30
                    state[i * 5 + 4] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
                    break
            seq_mode = len(tokens) > 1
            x = w['emb.weight'][tokens if seq_mode else tokens[0]]
            for i in range(args.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                dd = self.strategy[i]
                dev = dd.device
                atype = dd.atype
                wtype = dd.wtype
                if seq_mode:
                    if 'cuda' in str(dev) and os.environ['RWKV_CUDA_ON'] == '1':
                        ATT = self.cuda_att_seq if wtype != torch.uint8 else self.cuda_att_seq_i8
                    else:
                        ATT = self.att_seq if wtype != torch.uint8 else self.att_seq_i8
                    FFN = self.ffn_seq if wtype != torch.uint8 else self.ffn_seq_i8
                else:
                    ATT = self.att_one if wtype != torch.uint8 else self.att_one_i8
                    FFN = self.ffn_one if wtype != torch.uint8 else self.ffn_one_i8
                x = x.to(dtype=atype, device=dev)
                kw = w[f'{att}key.weight']
                vw = w[f'{att}value.weight']
                rw = w[f'{att}receptance.weight']
                ow = w[f'{att}output.weight']
                if dd.stream:
                    kw = kw.to(device=dev, non_blocking=True)
                    vw = vw.to(device=dev, non_blocking=True)
                    rw = rw.to(device=dev, non_blocking=True)
                    ow = ow.to(device=dev, non_blocking=True)
                kmx = w[f'{att}key.weight_mx'] if wtype == torch.uint8 else x
                krx = w[f'{att}key.weight_rx'] if wtype == torch.uint8 else x
                kmy = w[f'{att}key.weight_my'] if wtype == torch.uint8 else x
                kry = w[f'{att}key.weight_ry'] if wtype == torch.uint8 else x
                vmx = w[f'{att}value.weight_mx'] if wtype == torch.uint8 else x
                vrx = w[f'{att}value.weight_rx'] if wtype == torch.uint8 else x
                vmy = w[f'{att}value.weight_my'] if wtype == torch.uint8 else x
                vry = w[f'{att}value.weight_ry'] if wtype == torch.uint8 else x
                rmx = w[f'{att}receptance.weight_mx'] if wtype == torch.uint8 else x
                rrx = w[f'{att}receptance.weight_rx'] if wtype == torch.uint8 else x
                rmy = w[f'{att}receptance.weight_my'] if wtype == torch.uint8 else x
                rry = w[f'{att}receptance.weight_ry'] if wtype == torch.uint8 else x
                omx = w[f'{att}output.weight_mx'] if wtype == torch.uint8 else x
                orx = w[f'{att}output.weight_rx'] if wtype == torch.uint8 else x
                omy = w[f'{att}output.weight_my'] if wtype == torch.uint8 else x
                ory = w[f'{att}output.weight_ry'] if wtype == torch.uint8 else x
                x, state[i * 5 + 0], state[i * 5 + 1], state[i * 5 + 2], state[i * 5 + 3] = ATT(x, state[i * 5 + 0], state[i * 5 + 1], state[i * 5 + 2], state[i * 5 + 3], w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'], w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'], w[f'{att}time_decay'], w[f'{att}time_first'], kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory)
                return (state[0].tolist(), token_len)

    def generate(self, prompt: str, stop: Union[str, List[str], None]=None) -> Iterable[Tuple[str, str, int, int]]:
        import numpy as np
        quick_log(None, None, 'Generation Prompt:\n' + prompt)
        cache = None
        delta_prompt = prompt
        try:
            cache = state_cache.longest_prefix_state(state_cache.LongestPrefixStateBody(prompt=prompt), None)
        except HTTPException:
            pass
        if cache is None or cache['prompt'] == '' or cache['state'] is None:
            self.model_state = None
            self.model_tokens = []
        else:
            delta_prompt = prompt[len(cache['prompt']):]
            self.model_state = cache['state']
            self.model_tokens = cache['tokens']
            logits = cache['logits']
        prompt_token_len = 0
        if delta_prompt != '':
            logits, prompt_token_len = self.run_rnn(self.fix_tokens(self.pipeline.encode(delta_prompt)))
            try:
                state_cache.add_state(state_cache.AddStateBody(prompt=prompt, tokens=self.model_tokens, state=self.model_state, logits=logits))
            except HTTPException:
                pass
        begin = len(self.model_tokens)
        out_last = begin
        occurrence: Dict = {}
        completion_token_len = 0
        response = ''
        for i in range(self.max_tokens_per_generation):
            self.adjust_forward_logits(logits, occurrence, i)
            token = self.pipeline.sample_logits(logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
            if token == self.EOS_ID:
                try:
                    state_cache.add_state(state_cache.AddStateBody(prompt=prompt + response, tokens=self.model_tokens, state=self.model_state, logits=logits))
                except HTTPException:
                    pass
                yield (response, '', prompt_token_len, completion_token_len)
                break
            self.adjust_occurrence(occurrence, token)
            logits, _ = self.run_rnn([token])
            completion_token_len = completion_token_len + 1
            delta: str = self.delta_postprocess(self.pipeline.decode(self.model_tokens[out_last:]))
            if '�' not in delta:
                response += delta
                if stop is not None:
                    if type(stop) == str:
                        if stop in response:
                            try:
                                state_cache.add_state(state_cache.AddStateBody(prompt=prompt + response, tokens=self.model_tokens, state=self.model_state, logits=logits))
                            except HTTPException:
                                pass
                            response = response.split(stop)[0]
                            yield (response, '', prompt_token_len, completion_token_len)
                            break
                    elif type(stop) == list:
                        exit_flag = False
                        for s in stop:
                            if s in response:
                                try:
                                    state_cache.add_state(state_cache.AddStateBody(prompt=prompt + response, tokens=self.model_tokens, state=self.model_state, logits=logits))
                                except HTTPException:
                                    pass
                                exit_flag = True
                                response = response.split(s)[0]
                                yield (response, '', prompt_token_len, completion_token_len)
                                break
                        if exit_flag:
                            break
                out_last = begin + i + 1
                if i == self.max_tokens_per_generation - 1:
                    try:
                        state_cache.add_state(state_cache.AddStateBody(prompt=prompt + response, tokens=self.model_tokens, state=self.model_state, logits=logits))
                    except HTTPException:
                        pass
                yield (response, delta, prompt_token_len, completion_token_len)