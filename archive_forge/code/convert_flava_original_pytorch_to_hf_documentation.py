import argparse
import os
import torch
from transformers import FlavaConfig, FlavaForPreTraining
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint

    Copy/paste/tweak model's weights to transformers design.
    