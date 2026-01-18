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
def _create_heatmap(self, x_data: pd.Series, y_data: pd.Series, z_data: Optional[pd.Series]=None, color_data: Optional[pd.Series]=None, style: Optional[str]=None, interactive: bool=True) -> None:
    """
        Create a heatmap using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
    plt.figure(figsize=(10, 8))
    if style:
        sns.set_style(style)
    data_matrix = pd.pivot_table(pd.DataFrame({'x': x_data, 'y': y_data, 'color': color_data}), values='color', index='y', columns='x')
    sns.heatmap(data_matrix, cmap='viridis')
    plt.title('Heatmap')
    if interactive:
        mplcursors.cursor(hover=True)