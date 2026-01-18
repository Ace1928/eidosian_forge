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
def _decrypt(self, ciphertext, ref_key, media_info):
    length_kb = int(math.ceil(len(ciphertext) / 1024))
    progress = self._create_progress_iterator(range(length_kb), length_kb, 'Decrypt        ')
    try:
        plaintext = self._media_cipher.decrypt(ciphertext, ref_key, media_info)
        progress.update(length_kb)
        return plaintext
    except Exception as e:
        progress.set_description('Decrypt Error  ')
        logger.error(e)
    return None