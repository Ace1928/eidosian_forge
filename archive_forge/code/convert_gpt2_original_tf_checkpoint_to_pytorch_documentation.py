import argparse
import torch
from transformers import GPT2Config, GPT2Model, load_tf_weights_in_gpt2
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
Convert OpenAI GPT checkpoint.