from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class TelegramChatApiLoader(BaseLoader):
    """Load `Telegram` chat json directory dump."""

    def __init__(self, chat_entity: Optional[EntityLike]=None, api_id: Optional[int]=None, api_hash: Optional[str]=None, username: Optional[str]=None, file_path: str='telegram_data.json'):
        """Initialize with API parameters.

        Args:
            chat_entity: The chat entity to fetch data from.
            api_id: The API ID.
            api_hash: The API hash.
            username: The username.
            file_path: The file path to save the data to. Defaults to
                 "telegram_data.json".
        """
        self.chat_entity = chat_entity
        self.api_id = api_id
        self.api_hash = api_hash
        self.username = username
        self.file_path = file_path

    async def fetch_data_from_telegram(self) -> None:
        """Fetch data from Telegram API and save it as a JSON file."""
        from telethon.sync import TelegramClient
        data = []
        async with TelegramClient(self.username, self.api_id, self.api_hash) as client:
            async for message in client.iter_messages(self.chat_entity):
                is_reply = message.reply_to is not None
                reply_to_id = message.reply_to.reply_to_msg_id if is_reply else None
                data.append({'sender_id': message.sender_id, 'text': message.text, 'date': message.date.isoformat(), 'message.id': message.id, 'is_reply': is_reply, 'reply_to_id': reply_to_id})
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

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

    def _combine_message_texts(self, message_threads: Dict[int, List[int]], data: pd.DataFrame) -> str:
        """
        Combine the message texts for each parent message ID based             on the list of message threads.

        Args:
            message_threads (dict): A dictionary where the key is the parent message                 ID and the value is a list of message IDs in ascending order.
            data (pd.DataFrame): A DataFrame containing the conversation data:
                - message.sender_id
                - text
                - date
                - message.id
                - is_reply
                - reply_to_id

        Returns:
            str: A combined string of message texts sorted by date.
        """
        combined_text = ''
        for parent_id, message_ids in message_threads.items():
            message_texts = data[data['message.id'].isin(message_ids)].sort_values(by='date')['text'].tolist()
            message_texts = [str(elem) for elem in message_texts]
            combined_text += ' '.join(message_texts) + '.\n'
        return combined_text.strip()

    def load(self) -> List[Document]:
        """Load documents."""
        if self.chat_entity is not None:
            try:
                import nest_asyncio
                nest_asyncio.apply()
                asyncio.run(self.fetch_data_from_telegram())
            except ImportError:
                raise ImportError('`nest_asyncio` package not found.\n                    please install with `pip install nest_asyncio`\n                    ')
        p = Path(self.file_path)
        with open(p, encoding='utf8') as f:
            d = json.load(f)
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('`pandas` package not found. \n                please install with `pip install pandas`\n                ')
        normalized_messages = pd.json_normalize(d)
        df = pd.DataFrame(normalized_messages)
        message_threads = self._get_message_threads(df)
        combined_texts = self._combine_message_texts(message_threads, df)
        return text_to_docs(combined_texts)