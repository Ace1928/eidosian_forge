from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load data into Document objects.