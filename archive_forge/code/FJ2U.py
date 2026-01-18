#!/home/lloyd/indego/bin/python

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
    QApplication,
    QTreeView,
    QFileSystemModel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for file paths
SOURCE_DIR = Path("/home/lloyd/Dropbox/To_Sort")
DUPLICATES_DIR = Path("/home/lloyd/duplicates")
SORTED_DIR = Path("/home/lloyd/sorted")
ERRORS_DIR = Path("/home/lloyd/errors")
INDEX_FILE = Path("/home/lloyd/index.json")
ERROR_INDEX_FILE = Path("/home/lloyd/error_index.json")

# Ensure directories exist
for directory in [DUPLICATES_DIR, SORTED_DIR, ERRORS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist
for directory in [DUPLICATES_DIR, SORTED_DIR, ERRORS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class FileOperationWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(
        self,
        directory: Path,
        duplicates_index: dict,
        kept_files_index: dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.directory = directory
        self.duplicates_index = duplicates_index
        self.kept_files_index = kept_files_index

    async def calculate_file_hash(self, file_path: Path) -> str:
        BUF_SIZE = 65536  # Read in 64kb chunks
        sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                data = await f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

    async def move_file(self, file_path: Path, destination: Path) -> None:
        if not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)
        destination_file = destination / file_path.name
        shutil.move(str(file_path), str(destination_file))

    async def process_file(
        self,
        file_path: Path,
        file_hash: str,
        duplicates_index: Dict[str, List[str]],
        kept_files_index: Dict[str, str],
    ) -> None:
        """
        Processes a file by checking for duplicates and moving it to the appropriate directory.
        Implements advanced hashing and file management strategies.

        Args:
            file_path (Path): The path to the file to be processed.
            file_hash (str): The calculated hash of the file.
            duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
            kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
        """
        file_extension: str = file_path.suffix  # Extract file extension
        destination_dir: Path = SORTED_DIR / file_extension.strip(
            "."
        )  # Define destination directory based on file extension
        today_date: str = datetime.now().strftime(
            "_%d%m%y"
        )  # Get today's date in specified format
        standardized_name: str = (
            f"{file_hash}{today_date}{file_extension}"  # Construct standardized file name
        )
        destination_file: Path = (
            destination_dir / standardized_name
        )  # Define destination file path

        if file_hash in duplicates_index:
            await move_file(
                file_path, DUPLICATES_DIR
            )  # Move file to duplicates directory
            duplicates_index[file_hash].append(
                str(file_path)
            )  # Add file path to duplicates index
        else:
            duplicates_index[file_hash] = [
                str(file_path)
            ]  # Initialize list with file path in duplicates index
            kept_files_index[str(destination_file)] = str(
                file_path
            )  # Add file path to kept files index
            if not destination_dir.exists():
                destination_dir.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure destination directory exists
            shutil.move(
                str(file_path), str(destination_file)
            )  # Move file to destination

    async def process_directory(
        self,
        directory: Path,
        duplicates_index: Dict[str, List[str]],
        kept_files_index: Dict[str, str],
    ) -> None:
        """
        Recursively processes each file in a directory and its subdirectories.
        Employs advanced asynchronous programming techniques for efficiency.

        Args:
                    directory (Path): The path to the directory to be processed.
            duplicates_index (Dict[str, List[str]]): A dictionary to store duplicate file information.
            kept_files_index (Dict[str, str]): A dictionary to store information about kept files.
        """
        for item in directory.iterdir():  # Iterate over items in directory
            if item.is_file():  # Check if item is a file
                file_hash: str = await self.calculate_file_hash(item)  # Calculate file hash
                await self.process_file(
                    item, file_hash, duplicates_index, kept_files_index
                )  # Process file
            elif item.is_dir():  # Check if item is a directory
                await self.process_directory(
                    item, duplicates_index, kept_files_index
                )  # Recursively process directory

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.process_directory(
                self.directory, self.duplicates_index, self.kept_files_index
            )
        )
        loop.close()
        self.finished.emit()


class FileBrowserMainWindow(QMainWindow):
    """
    A comprehensive file browser main window that integrates a tree view and a progress bar for asynchronous file processing.
    This class is designed to provide a high-level interface for browsing files and directories, while also facilitating
    the asynchronous execution of file operations, such as moving or processing files, with real-time progress updates.
    """

    def __init__(self) -> None:
        """
        Initializes the FileBrowserMainWindow instance, setting up the UI components and preparing the environment for asynchronous file operations.
        """
        super().__init__()
        self.setWindowTitle("File Browser with Asynchronous Processing")
        self.resize(800, 600)
        self.tree_view = QTreeView()
        self.progress_bar = QProgressBar()
        self.init_ui()

    def init_ui(self) -> None:
        """
        Initializes the user interface components and layout of the main window.
        This method sets up the file system model for the tree view, configures the progress bar, and organizes the layout.
        """
        # Initialize the file system model with an empty root path to display all drives on the system
        self.model = QFileSystemModel()
        self.model.setRootPath('')

        # Configure the tree view to use the file system model
        self.tree_view.setModel(self.model)

        # Initialize the layout and add the tree view and progress bar
        layout = QVBoxLayout()
        widget = QWidget()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.progress_bar)
        widget.setLayout(layout)

        # Set the central widget of the main window to the layout container
        self.setCentralWidget(widget)

    def start_file_processing(self) -> None:
        """
        Initiates the file processing operation in a separate thread.
        This method creates a new thread and worker object for processing files asynchronously,
        connecting signals and slots to update the progress bar and manage thread lifecycle.
        """
        # Create a new thread for running file operations without blocking the UI
        self.thread = QThread()
        self.worker = FileOperationWorker(directory=Path("/path/to/directory"), duplicates_index={}, kept_files_index={})

        # Move the worker to the newly created thread
        self.worker.moveToThread(self.thread)

        # Connect signals and slots to start file operations and update the UI based on progress
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread, initiating asynchronous file operations
        self.thread.start()

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar value.

        Args:
            value (int): The progress value to set.
        """
        self.progress_bar.setValue(value)


def launch_file_browser(app: QApplication) -> Path:
    """
    Launches the file browser GUI, returning the selected directory path.

    Returns:
        Path: The selected directory path from the file browser GUI.
    """
    file_browser = FileBrowserMainWindow()
    file_browser.show()
    app.exec_()
    selected_indexes = file_browser.tree_view.selectedIndexes()
    if selected_indexes:
        selected_path = Path(file_browser.model.filePath(selected_indexes[0]))
        return selected_path
    return SOURCE_DIR




async def move_file(file_path: Path, destination: Path) -> None:
    """
    Moves a file to a specified destination directory, ensuring the destination exists.

    Args:
        file_path (Path): The path to the file to be moved.
        destination (Path): The destination directory path where the file should be moved.
    """
    if not destination.exists():
        destination.mkdir(
            parents=True, exist_ok=True
        )  # Ensure destination directory exists
    destination_file: Path = 
        destination / file_path.name
       # Define destination file path
    shutil.move(str(file_path), str(destination_file))  # Move file to destination


def run_asyncio_loop(loop: asyncio.AbstractEventLoop, coro: Coroutine) -> None:
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)


async def async_main(app: QApplication) -> None:
    """
    The main asynchronous function to be executed.
    """
    selected_directory = launch_file_browser(app)
    duplicates_index: Dict[str, List[str]] = {}
    kept_files_index: Dict[str, str] = {}
    await process_directory(selected_directory, duplicates_index, kept_files_index)


def is_admin() -> bool:
    """
    Determines if the script is running with administrator or superuser privileges across different operating systems,
    including Windows, Linux, and Android. It ensures that on Android, the script does not run as admin unless the user
    has superuser privileges, which is generally not common. If not running with the required privileges, it prompts the
    user to relaunch the program with elevated privileges.

    Returns:
        bool: True if the script has administrator privileges or is running as root/superuser, False otherwise.
    """
    try:
        if os.name == "nt":  # Checks if the operating system is Windows
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if not is_admin:
                logging.info("Requesting elevated privileges on Windows.")
                response = input(
                    "This script requires administrator privileges. Would you like to relaunch with elevated privileges? (y/n): "
                )
                if response.lower() == "y":
                    ctypes.windll.shell32.ShellExecuteW(
                        None, "runas", sys.executable, " ".join(sys.argv), None, 1
                    )
                    sys.exit(0)
            return is_admin
        elif (
            os.name == "posix"
        ):  # Checks if the operating system is POSIX-compliant (Linux, Unix, macOS, etc.)
            is_root = os.geteuid() == 0
            if not is_root:
                logging.info("Requesting superuser privileges on POSIX systems.")
                response = input(
                    "This script requires superuser privileges. Would you like to attempt to relaunch with sudo? (y/n): "
                )
                if response.lower() == "y":
                    os.execvp("sudo", ["sudo"] + sys.argv)
            return is_root
        elif os.name == "android":  # Specific check for Android operating system
            # Android does not typically allow applications to run with superuser privileges unless rooted
            # This check attempts to run a command that requires superuser privileges and checks the outcome
            is_superuser = os.system("su -c id") == 0
            if not is_superuser:
                logging.info(
                    "Superuser privileges are required on Android, but not available."
                )
            return is_superuser
        else:
            # For any other operating systems not explicitly checked, direct to support for upgrade/implementation
            logging.info(
                "Operating system not currently supported. Contact support for implementation."
            )
            return False
    except Exception as e:
        # Log any exceptions that occur during the check for administrative privileges
        logging.error(f"Error checking admin privileges: {e}")
        return False


def main():
    app = QApplication(sys.argv)
    mainWindow = FileBrowserMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
