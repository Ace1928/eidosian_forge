import bz2
import io
import json
import lzma
import os
import re
import requests
import subprocess
import zstandard as zstd
from bs4 import BeautifulSoup
from os.path import isfile
from os.path import join as pjoin
from time import sleep, time
from collections import defaultdict
from parlai.core.params import ParlaiParser
from data_utils import word_url_tokenize

    Set up args.
    