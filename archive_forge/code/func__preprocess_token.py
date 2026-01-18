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
def _preprocess_token(self, token):
    token = token.lower()
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    token = stemmer.stem(token)
    token = lemmatizer.lemmatize(token)
    return token