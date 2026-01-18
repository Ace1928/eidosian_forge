import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any, Optional
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes
class FileOperationWorker(QObject):
    """
    Asynchronously handles file operations including hashing, detecting duplicates,
    and organizing files into directories based on their hash values and extensions.

    Attributes:
        directory (Path): The root directory for file operations.
        duplicates_index (Dict[str, List[str]]): Tracks duplicate files by their hash.
        kept_files_index (Dict[str, str]): Tracks files that have been processed without duplication.
        progress (pyqtSignal): Signal to report operation progress.
        finished (pyqtSignal): Signal to indicate completion of operations.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, directory: Path, duplicates_index: Dict[str, List[str]], kept_files_index: Dict[str, str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.directory: Path = directory
        self.duplicates_index: Dict[str, List[str]] = duplicates_index
        self.kept_files_index: Dict[str, str] = kept_files_index

    async def calculate_file_hash(self, file_path: Path) -> str:
        """
        Asynchronously calculates the SHA-256 hash of a file.

        Args:
            file_path (Path): The path to the file.

        Returns:
            str: The hex digest of the file hash.
        """
        BUF_SIZE = 65536
        sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                data = await f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

    async def move_file(self, file_path: Path, destination: Path) -> None:
        """
        Asynchronously moves a file to a specified destination directory.

        Args:
            file_path (Path): The path to the file to be moved.
            destination (Path): The target directory path.
        """
        try:
            if not destination.exists():
                destination.mkdir(parents=True, exist_ok=True)
                destination_file = destination / file_path.name
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, os.rename, str(file_path), str(destination_file))
                logger.debug(f'Moved file: {file_path} to {destination_file}')
        except Exception as e:
            logger.error(f'Error moving file {file_path}: {e}')
            await self.move_file(file_path, ERRORS_DIR)

    async def process_file(self, file_path: Path, file_hash: str, duplicates_index: Dict[str, List[str]], kept_files_index: Dict[str, str]) -> None:
        """
        Processes a file by checking for duplicates and moving it to the appropriate directory.
        Implements advanced hashing and file management strategies.

        Args:
            file_path (Path): The path to the file to be processed.
            file_hash (str): The calculated hash of the file.
            duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
            kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
        """
        try:
            file_extension: str = file_path.suffix
            destination_dir: Path = SORTED_DIR / file_extension.strip('.')
            today_date: str = datetime.now().strftime('_%d%m%y')
            standardized_name: str = f'{file_hash}{today_date}{file_extension}'
            destination_file: Path = destination_dir / standardized_name
            if file_hash in duplicates_index:
                await self.move_file(file_path, DUPLICATES_DIR)
                duplicates_index[file_hash].append(str(file_path))
                logger.debug(f'Duplicate file found: {file_path}')
            else:
                duplicates_index[file_hash] = [str(file_path)]
                kept_files_index[str(destination_file)] = str(file_path)
                if not destination_dir.exists():
                    destination_dir.mkdir(parents=True, exist_ok=True)
                await self.move_file(file_path, destination_file)
                logger.debug(f'File processed: {file_path}')
        except Exception as e:
            logger.error(f'Error processing file {file_path}: {e}')
            await self.move_file(file_path, ERRORS_DIR)

    async def process_directory(self, directory: Path, duplicates_index: Dict[str, List[str]], kept_files_index: Dict[str, str]) -> None:
        """
        Recursively processes each file in a directory and its subdirectories.
        Employs advanced asynchronous programming techniques for efficiency.

        Args:
            directory (Path): The directory to process.
            duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
            kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
        """
        try:
            total_files = sum((1 for _ in directory.rglob('*')))
            processed_files = 0
            loop = asyncio.get_running_loop()
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_file():
                        file_path = Path(entry.path)
                        try:
                            file_hash = await self.calculate_file_hash(file_path)
                            await self.process_file(file_path, file_hash, duplicates_index, kept_files_index)
                        except Exception as e:
                            logger.error(f'Error processing file {file_path}: {e}')
                            await self.move_file(file_path, ERRORS_DIR)
                    elif entry.is_dir():
                        await self.process_directory(Path(entry.path), duplicates_index, kept_files_index)
            self.finished.emit()
        except Exception as e:
            logger.error(f'Error processing directory {directory}: {e}')

    async def run(self) -> None:
        """
        Initiates the file processing operation asynchronously.
        """
        try:
            await self.process_directory(self.directory, self.duplicates_index, self.kept_files_index)
        except Exception as e:
            logger.error(f'Error running file operations: {e}')