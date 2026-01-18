import json
import os
from typing import List, Union
import fire
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM  # @manual
Convert llama weights from huggingface format to consolidated format.
    params:
    model_path: model name or path to the model directory.
    model_size: Llama model size, one of 7B, 13B, 34B, 30B, 65B, 70B.
    output_dir: directory to save Llama weights, should contains params.json.
    