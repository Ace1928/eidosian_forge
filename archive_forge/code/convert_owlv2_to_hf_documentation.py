import argparse
import collections
import os
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging

    Copy/paste/tweak model's weights to our OWL-ViT structure.
    