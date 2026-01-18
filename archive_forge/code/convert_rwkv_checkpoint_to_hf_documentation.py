import argparse
import gc
import json
import os
import re
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint
Convert a RWKV checkpoint from BlinkDL to the Hugging Face format.