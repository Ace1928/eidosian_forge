import argparse
from collections import OrderedDict
from pathlib import Path
import torch
from transformers import (
from transformers.utils import logging

    Copy/paste/tweak model's weights to our VisualBERT structure.
    