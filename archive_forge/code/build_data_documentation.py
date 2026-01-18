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

        Performs a HEAD request to check if the URL / Google Drive ID is live.
        