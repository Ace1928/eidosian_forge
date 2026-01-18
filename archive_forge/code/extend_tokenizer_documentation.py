import os
import fire
import re
from transformers import LlamaTokenizer
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model

Code borrowed from https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py
