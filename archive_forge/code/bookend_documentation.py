import json
from typing import Any, List
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
Embed a query using a Bookend deployed embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        