from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build
import os
import random
import json
import numpy as np
import torch
import tqdm
def _setup_cands(self):
    self.fixed_cands = None
    self.fixed_cands_enc = None
    if self.fcp is not None:
        with open(self.fcp) as f:
            self.fixed_cands = [c.replace('\n', '') for c in f.readlines()]
        cands_enc_file = '{}.cands_enc'.format(self.fcp)
        print('loading saved cand encodings')
        if os.path.isfile(cands_enc_file):
            self.fixed_cands_enc = torch.load(cands_enc_file, map_location=lambda cpu, _: cpu)
        else:
            print('Extracting cand encodings')
            self.model.eval()
            pbar = tqdm.tqdm(total=len(self.fixed_cands), unit='cand', unit_scale=True, desc='Extracting candidate encodings')
            fixed_cands_enc = []
            for _, batch in enumerate([self.fixed_cands[i:i + 50] for i in range(0, len(self.fixed_cands) - 50, 50)]):
                embedding = self.model(None, None, batch)[1].detach()
                fixed_cands_enc.append(embedding)
                pbar.update(50)
            self.fixed_cands_enc = torch.cat(fixed_cands_enc, 0)
            torch.save(self.fixed_cands_enc, cands_enc_file)