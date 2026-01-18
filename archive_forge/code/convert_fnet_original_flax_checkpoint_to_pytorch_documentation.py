import argparse
import torch
from flax.training.checkpoints import restore_checkpoint
from transformers import FNetConfig, FNetForPreTraining
from transformers.utils import logging
Convert FNet checkpoint.