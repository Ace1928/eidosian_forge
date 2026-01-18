import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import math
import zipfile
import parlai.utils.logging as logging
def _naming_fn(url, url_metadata=None):
    return hashlib.md5(url.encode('utf-8')).hexdigest()