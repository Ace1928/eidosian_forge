import re
from pathlib import Path
from typing import Iterator, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Remove front matter metadata from the given content.