import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import os
import json
from collections import defaultdict, deque
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math
import tensorboard
from torch.utils.tensorboard import SummaryWriter
class RWKVRecurrence:

    def __init__(self, d_model):
        self.d_model = d_model
        self.rwkv_weights1 = np.random.randn(d_model, d_model)
        self.rwkv_weights2 = np.random.randn(d_model, d_model)
        self.rwkv_bias1 = np.zeros(d_model)
        self.rwkv_bias2 = np.zeros(d_model)

    def forward(self, x, state):
        if state is None:
            state = np.zeros((x.shape[0], self.d_model))
        x = np.dot(x, self.rwkv_weights1) + self.rwkv_bias1
        state = np.dot(state, self.rwkv_weights2) + self.rwkv_bias2
        state = np.maximum(x, state)
        return (x, state)