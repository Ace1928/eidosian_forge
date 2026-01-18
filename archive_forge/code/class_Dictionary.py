import argparse
import json
import os
import re
import shutil
import torch
from transformers import BioGptConfig, BioGptForCausalLM
from transformers.models.biogpt.tokenization_biogpt import VOCAB_FILES_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import WEIGHTS_NAME, logging
class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(self, *, bos='<s>', pad='<pad>', eos='</s>', unk='<unk>', extra_special_symbols=None):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = (bos, unk, pad, eos)
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and (not overwrite):
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def _load_meta(self, lines):
        return 0

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, 'r', encoding='utf-8') as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception('Incorrect encoding detected in {}, please rebuild the dataset'.format(f))
            return
        lines = f.readlines()
        indices_start_line = self._load_meta(lines)
        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(' ', 1)
                if field == '#fairseq:overwrite':
                    overwrite = True
                    line, field = line.rsplit(' ', 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and (not overwrite):
                    raise RuntimeError("Duplicate word found when loading Dictionary: '{}'. Duplicate words can overwrite earlier ones by adding the #fairseq:overwrite flag at the end of the corresponding row in the dictionary file. If using the Camembert model, please download an updated copy of the model file.".format(word))
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt> [flags]'")