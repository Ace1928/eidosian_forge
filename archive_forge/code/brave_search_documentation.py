import json
from typing import List
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
Query the Brave search engine and return the results as a list of Documents.

        Args:
            query: The query to search for.

        Returns: The results as a list of Documents.

        