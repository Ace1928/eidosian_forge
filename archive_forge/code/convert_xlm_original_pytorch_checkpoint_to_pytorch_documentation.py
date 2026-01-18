import argparse
import json
import numpy
import torch
from transformers.models.xlm.tokenization_xlm import VOCAB_FILES_NAMES
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
Convert OpenAI GPT checkpoint.