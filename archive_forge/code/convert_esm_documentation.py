import argparse
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory
import esm as esm_module
import torch
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences
from esm.esmfold.v1.pretrained import esmfold_v1
from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig
from transformers.models.esm.modeling_esm import (
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
from transformers.models.esm.tokenization_esm import EsmTokenizer
from transformers.utils import logging

    Copy/paste/tweak esm's weights to our BERT structure.
    