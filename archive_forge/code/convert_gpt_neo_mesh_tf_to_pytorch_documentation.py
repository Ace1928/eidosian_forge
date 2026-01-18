import argparse
import json
from transformers import GPTNeoConfig, GPTNeoForCausalLM, load_tf_weights_in_gpt_neo
from transformers.utils import logging
Convert GPT Neo checkpoint.