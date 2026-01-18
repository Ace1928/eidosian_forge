from abc import ABCMeta
from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Optional
from selenium.types import AnyKey
from selenium.webdriver.common.utils import keys_to_typing
class LocalFileDetector(FileDetector):
    """Detects files on the local disk."""

    def is_local_file(self, *keys: AnyKey) -> Optional[str]:
        file_path = ''.join(keys_to_typing(keys))
        with suppress(OSError):
            if Path(file_path).is_file():
                return file_path