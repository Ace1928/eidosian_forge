import argparse
import torch
from transformers import ImageGPTConfig, ImageGPTForCausalLM, load_tf_weights_in_imagegpt
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
Convert OpenAI Image GPT checkpoints.