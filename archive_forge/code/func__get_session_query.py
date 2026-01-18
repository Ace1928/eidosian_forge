from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
def _get_session_query(self, use_chat_handle_table: bool) -> str:
    joins_w_chat_handle = '\n            JOIN chat_handle_join ON\n                 chat_message_join.chat_id = chat_handle_join.chat_id\n            JOIN handle ON\n                 handle.ROWID = chat_handle_join.handle_id'
    joins_no_chat_handle = '\n            JOIN handle ON message.handle_id = handle.ROWID\n        '
    joins = joins_w_chat_handle if use_chat_handle_table else joins_no_chat_handle
    return f'\n            SELECT  message.date,\n                    handle.id,\n                    message.text,\n                    message.is_from_me,\n                    message.attributedBody\n            FROM message\n            JOIN chat_message_join ON\n                 message.ROWID = chat_message_join.message_id\n            {joins}\n            WHERE chat_message_join.chat_id = ?\n            ORDER BY message.date ASC;\n        '