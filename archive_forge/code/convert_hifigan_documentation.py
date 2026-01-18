import argparse
import numpy as np
import torch
from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig, logging
Convert SpeechT5 HiFi-GAN checkpoint.