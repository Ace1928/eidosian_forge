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

        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        