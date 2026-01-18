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
@dataclass
class EsmForProteinFoldingOutput(ModelOutput):
    """
    Output type of [`EsmForProteinFoldingOutput`].

    Args:
        frames (`torch.FloatTensor`):
            Output frames.
        sidechain_frames (`torch.FloatTensor`):
            Output sidechain frames.
        unnormalized_angles (`torch.FloatTensor`):
            Predicted unnormalized backbone and side chain torsion angles.
        angles (`torch.FloatTensor`):
            Predicted backbone and side chain torsion angles.
        positions (`torch.FloatTensor`):
            Predicted positions of the backbone and side chain atoms.
        states (`torch.FloatTensor`):
            Hidden states from the protein folding trunk.
        s_s (`torch.FloatTensor`):
            Per-residue embeddings derived by concatenating the hidden states of each layer of the ESM-2 LM stem.
        s_z (`torch.FloatTensor`):
            Pairwise residue embeddings.
        distogram_logits (`torch.FloatTensor`):
            Input logits to the distogram used to compute residue distances.
        lm_logits (`torch.FloatTensor`):
            Logits output by the ESM-2 protein language model stem.
        aatype (`torch.FloatTensor`):
            Input amino acids (AlphaFold2 indices).
        atom14_atom_exists (`torch.FloatTensor`):
            Whether each atom exists in the atom14 representation.
        residx_atom14_to_atom37 (`torch.FloatTensor`):
            Mapping between atoms in the atom14 and atom37 representations.
        residx_atom37_to_atom14 (`torch.FloatTensor`):
            Mapping between atoms in the atom37 and atom14 representations.
        atom37_atom_exists (`torch.FloatTensor`):
            Whether each atom exists in the atom37 representation.
        residue_index (`torch.FloatTensor`):
            The index of each residue in the protein chain. Unless internal padding tokens are used, this will just be
            a sequence of integers from 0 to `sequence_length`.
        lddt_head (`torch.FloatTensor`):
            Raw outputs from the lddt head used to compute plddt.
        plddt (`torch.FloatTensor`):
            Per-residue confidence scores. Regions of low confidence may indicate areas where the model's prediction is
            uncertain, or where the protein structure is disordered.
        ptm_logits (`torch.FloatTensor`):
            Raw logits used for computing ptm.
        ptm (`torch.FloatTensor`):
            TM-score output representing the model's high-level confidence in the overall structure.
        aligned_confidence_probs (`torch.FloatTensor`):
            Per-residue confidence scores for the aligned structure.
        predicted_aligned_error (`torch.FloatTensor`):
            Predicted error between the model's prediction and the ground truth.
        max_predicted_aligned_error (`torch.FloatTensor`):
            Per-sample maximum predicted error.
    """
    frames: torch.FloatTensor = None
    sidechain_frames: torch.FloatTensor = None
    unnormalized_angles: torch.FloatTensor = None
    angles: torch.FloatTensor = None
    positions: torch.FloatTensor = None
    states: torch.FloatTensor = None
    s_s: torch.FloatTensor = None
    s_z: torch.FloatTensor = None
    distogram_logits: torch.FloatTensor = None
    lm_logits: torch.FloatTensor = None
    aatype: torch.FloatTensor = None
    atom14_atom_exists: torch.FloatTensor = None
    residx_atom14_to_atom37: torch.FloatTensor = None
    residx_atom37_to_atom14: torch.FloatTensor = None
    atom37_atom_exists: torch.FloatTensor = None
    residue_index: torch.FloatTensor = None
    lddt_head: torch.FloatTensor = None
    plddt: torch.FloatTensor = None
    ptm_logits: torch.FloatTensor = None
    ptm: torch.FloatTensor = None
    aligned_confidence_probs: torch.FloatTensor = None
    predicted_aligned_error: torch.FloatTensor = None
    max_predicted_aligned_error: torch.FloatTensor = None