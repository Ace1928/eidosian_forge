import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import torch
import yaml
from tokenizers import Tokenizer
from transformers import OlmoConfig, OlmoForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers import OlmoForCausalLM, AutoTokenizer
def _write_tokenizer(output_path: Path, config: OlmoConfig, input_tokenizer_path: Path, fix_eos_token_id: bool=True) -> None:
    print(f'Saving a {GPTNeoXTokenizerFast.__name__} to {output_path}.')
    base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))
    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id
    if fix_eos_token_id and eos_token_id == 0:
        print('Changing eos_token_id from 0 to 50279.')
        eos_token_id = 50279
    tokenizer = GPTNeoXTokenizerFast(tokenizer_object=base_tokenizer, eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False), pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False), unk_token=None, bos_token=None)
    tokenizer.save_pretrained(output_path)