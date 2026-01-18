import argparse
import os
import align
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image
from tokenizer import Tokenizer
from transformers import (
from transformers.utils import logging

    Copy/paste/tweak model's weights to our ALIGN structure.
    