import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldingTrunk(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        c_s = config.sequence_state_dim
        c_z = config.pairwise_state_dim
        self.pairwise_positional_embedding = EsmFoldRelativePosition(config)
        self.blocks = nn.ModuleList([EsmFoldTriangularSelfAttentionBlock(config) for _ in range(config.num_blocks)])
        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0].detach().zero_()
        self.structure_module = EsmFoldStructureModule(config.structure_module)
        self.trunk2sm_s = nn.Linear(c_s, config.structure_module.sequence_dim)
        self.trunk2sm_z = nn.Linear(c_z, config.structure_module.pairwise_dim)
        self.chunk_size = config.chunk_size

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        """
        Inputs:
          seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
          x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """
        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats
        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            if no_recycles < 0:
                raise ValueError('Number of recycles must not be negative.')
            no_recycles += 1

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return (s, z)
        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)
        for recycle_idx in range(no_recycles):
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)
                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)
                structure = self.structure_module({'single': self.trunk2sm_s(s_s), 'pair': self.trunk2sm_z(s_z)}, true_aa, mask.float())
                recycle_s = s_s
                recycle_z = s_z
                recycle_bins = EsmFoldingTrunk.distogram(structure['positions'][-1][:, :, :3], 3.375, 21.375, self.recycle_bins)
        structure['s_s'] = s_s
        structure['s_z'] = s_z
        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        boundaries = torch.linspace(min_bin, max_bin, num_bins - 1, device=coords.device)
        boundaries = boundaries ** 2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        bins = torch.sum(dists > boundaries, dim=-1)
        return bins