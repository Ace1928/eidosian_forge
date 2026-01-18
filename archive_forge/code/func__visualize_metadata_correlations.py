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
def _visualize_metadata_correlations(self, metadata: Dict[str, Any], style: Optional[str]=None, interactive: bool=True, save_path: Optional[str]=None) -> None:
    """
        Create a heatmap for the correlation between numeric metadata fields.

        Args:
            metadata (Dict[str, Any]): The metadata to visualize.
            style (Optional[str]): The style theme for the charts.
            interactive (bool): Whether to enable interactive plot controls.
            save_path (Optional[str]): The path to save the visualizations, if desired.

        Returns:
            None
        """
    numeric_metadata = pd.DataFrame({k: v for k, v in metadata.items() if isinstance(v, (int, float))}).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_metadata, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Metadata Fields')
    if interactive:
        mplcursors.cursor(hover=True)
    if save_path:
        save_path_numeric_metadata_corr = f'{save_path}_numeric_metadata_corr.png'
        self._save_visualization(save_path_numeric_metadata_corr)