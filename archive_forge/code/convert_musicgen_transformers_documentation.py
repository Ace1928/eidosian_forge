import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple
import torch
from audiocraft.models import MusicGen
from transformers import (
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
from transformers.utils import logging
Function that takes the fairseq Musicgen state dict and renames it according to the HF
    module names. It further partitions the state dict into the decoder (LM) state dict, and that for the
    encoder-decoder projection.