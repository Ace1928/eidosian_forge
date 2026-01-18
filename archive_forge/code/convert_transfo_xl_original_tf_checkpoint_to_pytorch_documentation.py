import argparse
import os
import pickle
import sys
import torch
from transformers import TransfoXLConfig, TransfoXLLMHeadModel, load_tf_weights_in_transfo_xl
from transformers.models.deprecated.transfo_xl import tokenization_transfo_xl as data_utils
from transformers.models.deprecated.transfo_xl.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
Convert Transformer XL checkpoint and datasets.