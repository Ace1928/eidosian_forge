from __future__ import annotations
import math
from typing import List, NamedTuple, Union
import torch
import torchaudio
import torchaudio.lib.pybind11_prefixctc as cuctc
def _get_vocab_list(vocab_file):
    vocab = []
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            vocab.append(line[0])
    return vocab