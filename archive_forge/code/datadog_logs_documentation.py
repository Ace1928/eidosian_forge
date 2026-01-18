from datetime import datetime, timedelta
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

        Get logs from Datadog.

        Returns:
            A list of Document objects.
                - page_content
                - metadata
                    - id
                    - service
                    - status
                    - tags
                    - timestamp
        