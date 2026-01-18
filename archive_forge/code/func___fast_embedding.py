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