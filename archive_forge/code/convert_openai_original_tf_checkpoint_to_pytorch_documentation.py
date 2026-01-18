import argparse
import torch
from transformers import OpenAIGPTConfig, OpenAIGPTModel, load_tf_weights_in_openai_gpt
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
Convert OpenAI GPT checkpoint.