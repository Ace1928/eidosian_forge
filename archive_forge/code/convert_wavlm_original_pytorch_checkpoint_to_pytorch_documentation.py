import argparse
import torch
from unilm.wavlm.WavLM import WavLM as WavLMOrig
from unilm.wavlm.WavLM import WavLMConfig as WavLMConfigOrig
from transformers import WavLMConfig, WavLMModel, logging
Convert WavLM checkpoint.