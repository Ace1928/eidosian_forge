import argparse
import json
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
Convert VideoMAE checkpoints from the original repository: https://github.com/MCG-NJU/VideoMAE