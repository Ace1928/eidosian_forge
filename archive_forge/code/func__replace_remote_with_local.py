import glob
import logging
import os
import re
from typing import List, Tuple
import requests
def _replace_remote_with_local(file_path: str, docs_folder: str, pairs_url_path: List[Tuple[str, str]]) -> None:
    """Replace all URL with local files in a given file.

    Args:
        file_path: file for replacement
        docs_folder: the location of docs related to the project root
        pairs_url_path: pairs of URL and local file path to be swapped

    """
    relt_path = os.path.dirname(file_path).replace(docs_folder, '')
    depth = len([p for p in relt_path.split(os.path.sep) if p])
    with open(file_path, encoding='UTF-8') as fopen:
        body = fopen.read()
    for url, fpath in pairs_url_path:
        if depth:
            path_up = ['..'] * depth
            fpath = os.path.join(*path_up, fpath)
        body = body.replace(url, fpath)
    with open(file_path, 'w', encoding='UTF-8') as fw:
        fw.write(body)