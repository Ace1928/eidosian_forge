import json
import logging
from pathlib import Path
from typing import Iterator, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
Lazy loads the chat data from the folder.

        Yields:
            ChatSession: A chat session containing the loaded messages.

        