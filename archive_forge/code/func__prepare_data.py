import asyncio
import cProfile
import hashlib
import io
import itertools
import json
import logging
import resource
import os
import pstats
import queue
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from functools import reduce
from logging.handlers import MemoryHandler, RotatingFileHandler
from logging import StreamHandler, FileHandler
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator, Union, Callable
import coloredlogs
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import (
from wordcloud import WordCloud
from tqdm import tqdm
import requests
from transformers import BertModel, BertTokenizer
import functools
def _prepare_data(self, x_axis: str, y_axis: str, z_axis: Optional[str]=None, color: Optional[str]=None) -> Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
        Prepare the data for visualization by extracting the specified columns from the processed data.

        Args:
            x_axis (str): The data column to use for the x-axis.
            y_axis (str): The data column to use for the y-axis.
            z_axis (Optional[str]): The data column to use for the z-axis (for 3D plots).
            color (Optional[str]): The data column to use for color encoding.

        Returns:
            Tuple[pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]: The extracted data columns for visualization.
        """
    x_data = self.data[x_axis]
    y_data = self.data[y_axis]
    z_data = self.data[z_axis] if z_axis else None
    color_data = self.data[color] if color else None
    return (x_data, y_data, z_data, color_data)