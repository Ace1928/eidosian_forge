import argparse
import collections
import numpy as np
import torch
from flax import traverse_util
from t5x import checkpoints
from transformers import MT5Config, UMT5EncoderModel, UMT5ForConditionalGeneration
from transformers.utils import logging
Loads the config and model, converts the T5X checkpoint, and saves a PyTorch checkpoint.