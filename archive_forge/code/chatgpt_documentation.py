import datetime
import json
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Initialize a class object.

        Args:
            log_file: Path to the log file
            num_logs: Number of logs to load. If 0, load all logs.
        