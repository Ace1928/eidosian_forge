import argparse
import os
from pathlib import Path
import fairseq
import torch
from packaging import version
from torch import nn
from transformers import (
from transformers.utils import logging

    Copy/paste/tweak model's weights to our BERT structure.
    