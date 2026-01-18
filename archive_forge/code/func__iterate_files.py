import logging
import os
import re
import zipfile
from typing import Iterator, List, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _iterate_files(self, path: str) -> Iterator[str]:
    """Iterate over the files in a directory or zip file.

        Args:
            path (str): Path to the directory or zip file.

        Yields:
            str: The path to each file.
        """
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    yield os.path.join(root, file)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zip_file:
            for file in zip_file.namelist():
                if file.endswith('.txt'):
                    yield zip_file.extract(file)