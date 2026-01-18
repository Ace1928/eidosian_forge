import json
import logging
from pathlib import Path
from typing import Iterator, Union
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from langchain_community.chat_loaders.base import BaseChatLoader
class FolderFacebookMessengerChatLoader(BaseChatLoader):
    """Load `Facebook Messenger` chat data from a folder.

    Args:
        path (Union[str, Path]): The path to the directory
            containing the chat files.

    Attributes:
        path (Path): The path to the directory containing the chat files.

    """

    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__()
        self.directory_path = Path(path) if isinstance(path, str) else path

    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy loads the chat data from the folder.

        Yields:
            ChatSession: A chat session containing the loaded messages.

        """
        inbox_path = self.directory_path / 'inbox'
        for _dir in inbox_path.iterdir():
            if _dir.is_dir():
                for _file in _dir.iterdir():
                    if _file.suffix.lower() == '.json':
                        file_loader = SingleFileFacebookMessengerChatLoader(path=_file)
                        for result in file_loader.lazy_load():
                            yield result