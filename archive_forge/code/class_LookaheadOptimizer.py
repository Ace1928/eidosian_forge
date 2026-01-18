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
class LookaheadOptimizer:

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.fast_weights = None

    async def update_weights(self, model, gradients):
        if self.fast_weights is None:
            self.fast_weights = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        await self.base_optimizer.update_weights(model, gradients)
        if self.base_optimizer.state[next(iter(model.parameters()))]['step'] % self.k == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.fast_weights[name].mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)
                    param.data.copy_(self.fast_weights[name])