from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_community.document_loaders.base_o365 import (
from langchain_community.document_loaders.parsers.registry import get_parser
def _get_folder_from_path(self, drive: Drive) -> Union[Folder, Drive]:
    """
        Returns the folder or drive object located at the
        specified path relative to the given drive.

        Args:
            drive (Drive): The root drive from which the folder path is relative.

        Returns:
            Union[Folder, Drive]: The folder or drive object
            located at the specified path.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
    subfolder_drive = drive
    if self.folder_path is None:
        return subfolder_drive
    subfolders = [f for f in self.folder_path.split('/') if f != '']
    if len(subfolders) == 0:
        return subfolder_drive
    items = subfolder_drive.get_items()
    for subfolder in subfolders:
        try:
            subfolder_drive = list(filter(lambda x: subfolder in x.name, items))[0]
            items = subfolder_drive.get_items()
        except (IndexError, AttributeError):
            raise FileNotFoundError('Path {} not exist.'.format(self.folder_path))
    return subfolder_drive