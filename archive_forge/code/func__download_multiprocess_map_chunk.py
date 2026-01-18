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
def _download_multiprocess_map_chunk(pool_tup):
    """
    Helper function for Pool imap_unordered.

    Apparently function must be pickable (which apparently means must be
    defined at the top level of a module and can't be a lamdba) to be used in
    imap_unordered. Has to do with how it's passed to the subprocess.

    :param pool_tup: is a tuple where first arg is an array of tuples of url
    and dest file name for the current chunk and second arg is function to be
    called.
    :return: an array of tuples
    """
    items = pool_tup[0]
    path = pool_tup[1]
    fn = pool_tup[2]
    return [fn(it[0], path, it[1]) for it in items]