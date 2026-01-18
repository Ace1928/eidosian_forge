import argparse
from pathlib import Path
import torch
from transformers import OPTConfig, OPTModel
from transformers.utils import logging

    Copy/paste/tweak model's weights to our BERT structure.
    