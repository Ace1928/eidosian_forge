import argparse
import json
import tempfile
import torch
from huggingface_hub import hf_hub_download
from transformers import VitsConfig, VitsModel, VitsTokenizer, logging

    Copy/paste/tweak model's weights to transformers design.
    