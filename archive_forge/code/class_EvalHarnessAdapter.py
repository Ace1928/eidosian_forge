import os, sys, types, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from lm_eval import tasks, evaluator
from lm_eval.models.gpt2 import GPT2LM
class EvalHarnessAdapter(GPT2LM):

    def __init__(self):
        self.tokenizer = TokenizerWrapper(pipeline.tokenizer)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf
        res = []
        for COUNTER in range(len(requests)):
            n = COUNTER
            raw_src = requests[n][0][0] + requests[n][0][1]
            src = requests[n][1] + requests[n][2]
            raw_src = '\n' + raw_src
            src = RWKV_PAD + src
            sss = str(src)
            correct = True
            if sss in logitBuf:
                logit = logitBuf[sss]
                correct = correctBuf[sss]
            else:
                q_len = len(requests[n][1])
                q_len += len(RWKV_PAD)
                logit = 0
                with torch.no_grad():
                    outputs, _ = model.forward(src, None, full_output=True)
                    for i in range(q_len - 1, len(src) - 1):
                        oo = outputs[i].detach().float()
                        dst = src[i + 1]
                        logit += math.log(F.softmax(oo, dim=-1)[dst])
                        _, s_index = torch.sort(oo, descending=True)
                        pred = s_index[0].item()
                        if pred != dst:
                            correct = False
                    outputs = None
                    pred = None
                logitBuf[sss] = logit
                correctBuf[sss] = correct
            res += [(logit, correct)]
            if n % 1000 == 0:
                print(f'{n // 1000}/{len(requests) // 1000}', end=' ', flush=True)
        return res

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        results = evaluator.evaluate(lm=self, task_dict=tasks.get_task_dict(eval_tasks), provide_description=False, num_fewshot=num_fewshot, limit=None, bootstrap_iters=bootstrap_iters)
        return results