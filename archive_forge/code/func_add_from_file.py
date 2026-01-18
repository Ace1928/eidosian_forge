import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def add_from_file(self, f):
    """
        Loads a pre-existing dictionary from a text file and adds its symbols to this instance.
        """
    if isinstance(f, str):
        try:
            with open(f, 'r', encoding='utf-8') as fd:
                self.add_from_file(fd)
        except FileNotFoundError as fnfe:
            raise fnfe
        except UnicodeError:
            raise Exception(f'Incorrect encoding detected in {f}, please rebuild the dataset')
        return
    lines = f.readlines()
    for lineTmp in lines:
        line = lineTmp.strip()
        idx = line.rfind(' ')
        if idx == -1:
            raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
        word = line[:idx]
        self.encoder[word] = len(self.encoder)