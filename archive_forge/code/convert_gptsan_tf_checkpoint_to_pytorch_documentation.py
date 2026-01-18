import argparse
import json
import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import torch
Convert GPTSANJapanese checkpoints from the original repository to pytorch model.