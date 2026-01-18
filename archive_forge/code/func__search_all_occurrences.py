import glob
import logging
import os
import re
from typing import List, Tuple
import requests
def _search_all_occurrences(list_files: List[str], pattern: str) -> List[str]:
    """Search for all occurrences of specific pattern in a collection of files.

    Args:
        list_files: list of files to be scanned
        pattern: pattern for search, reg. expression

    """
    collected = []
    for file_path in list_files:
        with open(file_path, encoding='UTF-8') as fopem:
            body = fopem.read()
        found = re.findall(pattern, body)
        collected += found
    return collected