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
def _create_3d_plot(self, x_data: pd.Series, y_data: pd.Series, z_data: pd.Series, color_data: Optional[pd.Series]=None, style: Optional[str]=None, interactive: bool=True) -> None:
    """
        Create a 3D plot using the provided data and customization options.

        Args:
            x_data (pd.Series): The data for the x-axis.
            y_data (pd.Series): The data for the y-axis.
            z_data (pd.Series): The data for the z-axis.
            color_data (Optional[pd.Series]): The data for color encoding.
            style (Optional[str]): The style theme for the chart.
            interactive (bool): Whether to enable interactive plot controls.

        Returns:
            None
        """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if style:
        sns.set_style(style)
    ax.scatter(x_data, y_data, z_data, c=color_data, cmap='viridis')
    ax.set_title('3D Plot')
    ax.set_xlabel(x_data.name)
    ax.set_ylabel(y_data.name)
    ax.set_zlabel(z_data.name)
    if interactive:
        mplcursors.cursor(hover=True)