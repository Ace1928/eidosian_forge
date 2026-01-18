from yowsup.layers.protocol_media.mediacipher import MediaCipher
from yowsup.layers.protocol_media.protocolentities \
import threading
from tqdm import tqdm
import requests
import logging
import math
import sys
import os
import base64
def _create_progress_iterator(self, iterable, niterations, desc):
    return tqdm(iterable, total=niterations, unit='KB', dynamic_ncols=True, unit_scale=True, leave=True, desc=desc, ascii=True)