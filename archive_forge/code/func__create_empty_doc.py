from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
def _create_empty_doc(self) -> None:
    """Creates or replaces a document for this message history with no
        messages"""
    self.client.Documents.add_documents(collection=self.collection, workspace=self.workspace, data=[{'_id': self.session_id, self.messages_key: []}])