from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_message_threads(self, data: pd.DataFrame) -> dict:
    """Create a dictionary of message threads from the given data.

        Args:
            data (pd.DataFrame): A DataFrame containing the conversation                 data with columns:
                - message.sender_id
                - text
                - date
                - message.id
                - is_reply
                - reply_to_id

        Returns:
            dict: A dictionary where the key is the parent message ID and                 the value is a list of message IDs in ascending order.
        """

    def find_replies(parent_id: int, reply_data: pd.DataFrame) -> List[int]:
        """
            Recursively find all replies to a given parent message ID.

            Args:
                parent_id (int): The parent message ID.
                reply_data (pd.DataFrame): A DataFrame containing reply messages.

            Returns:
                list: A list of message IDs that are replies to the parent message ID.
            """
        direct_replies = reply_data[reply_data['reply_to_id'] == parent_id]['message.id'].tolist()
        all_replies = []
        for reply_id in direct_replies:
            all_replies += [reply_id] + find_replies(reply_id, reply_data)
        return all_replies
    parent_messages = data[~data['is_reply']]
    reply_messages = data[data['is_reply']].dropna(subset=['reply_to_id'])
    reply_messages['reply_to_id'] = reply_messages['reply_to_id'].astype(int)
    message_threads = {parent_id: [parent_id] + find_replies(parent_id, reply_messages) for parent_id in parent_messages['message.id']}
    return message_threads