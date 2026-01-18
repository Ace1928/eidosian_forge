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
class FileBrowserMainWindow(QMainWindow):
    """
    The main window for the file browser application.
    Displays a tree view of the file system and a progress bar for file operations.
    """
    thread: Optional[QThread] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle('File Browser')
        self.setGeometry(100, 100, 800, 600)
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setAnimated(False)
        self.tree_view.setIndentation(20)
        self.tree_view.setSortingEnabled(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout = QVBoxLayout()
        widget = QWidget()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.progress_bar)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def start_file_processing(self) -> None:
        """
        Initiates the file processing operation in a separate thread.
        This method creates a new thread and worker object for processing files asynchronously,
        connecting signals and slots to update the progress bar and manage thread lifecycle.
        """
        try:
            self.thread = QThread()
            self.worker = FileOperationWorker(directory=Path('/path/to/directory'), duplicates_index={}, kept_files_index={})
            self.worker.moveToThread(self.thread)

            def start_async():
                asyncio.create_task(self.worker.run())
            self.thread.started.connect(start_async)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
        except Exception as e:
            logger.error(f'Error starting file processing: {e}')

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar value.

        Args:
            value (int): The progress value to set.
        """
        try:
            self.progress_bar.setValue(value)
        except Exception as e:
            logger.error(f'Error updating progress bar: {e}')