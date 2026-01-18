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
def _create_unique_filepath(self, filepath):
    file_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    result_filename = filename
    dissected = os.path.splitext(filename)
    count = 0
    while os.path.exists(os.path.join(file_dir, result_filename)):
        count += 1
        result_filename = '%s_%d%s' % (dissected[0], count, dissected[1])
    return os.path.join(file_dir, result_filename)