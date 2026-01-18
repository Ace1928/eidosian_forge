import json
import logging
import time
from typing import Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Makes a call to Cube's REST API metadata endpoint.

        Returns:
            A list of documents with attributes:
                - page_content=column_title + column_description
                - metadata
                    - table_name
                    - column_name
                    - column_data_type
                    - column_member_type
                    - column_title
                    - column_description
                    - column_values
                    - cube_data_obj_type
        