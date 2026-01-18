"""
Git integration for Eidosian Forge.

Provides versioned memory storage through Git, allowing the system to track
its own evolution and maintain history of thoughts and developments.
"""

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GitMemoryManager:
    """Manages versioned memory storage using Git."""

    def __init__(
        self,
        repo_path: str,
        auto_commit: bool = True,
        commit_interval_minutes: int = 30,
    ):
        """
        Initialize Git memory manager.

        Args:
            repo_path: Path to Git repository
            auto_commit: Whether to automatically commit changes
            commit_interval_minutes: Interval between auto-commits in minutes
        """
        self.repo_path = Path(repo_path).absolute()
        self.auto_commit = auto_commit
        self.commit_interval = commit_interval_minutes * 60  # Convert to seconds
        self._initialize_repo()
        self._stop_event = threading.Event()

        if auto_commit:
            self._start_auto_commit_thread()

    def _initialize_repo(self) -> None:
        """Initialize Git repository if it doesn't exist."""
        if not self.repo_path.exists():
            logger.info(f"Creating Git repository at {self.repo_path}")
            os.makedirs(self.repo_path, exist_ok=True)

            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self.repo_path),
                    check=True,
                    capture_output=True,
                )

                # Create initial README
                readme_path = self.repo_path / "README.md"
                with open(readme_path, "w") as f:
                    f.write("# Eidosian Forge Memory Repository\n\n")
                    f.write(
                        "This repository contains the versioned memory of the Eidosian Forge AI system.\n"
                    )
                    f.write(f"Created: {datetime.now().isoformat()}\n")

                # Create directory structure
                memory_dirs = [
                    "thoughts",
                    "reflections",
                    "tasks",
                    "knowledge",
                    "changes",
                ]
                for d in memory_dirs:
                    os.makedirs(self.repo_path / d, exist_ok=True)
                    # Add .gitkeep to ensure directories are tracked
                    with open(self.repo_path / d / ".gitkeep", "w") as f:
                        pass

                # Initial commit
                self.commit("Initialize Eidosian Forge memory repository")
                logger.info("Git repository initialized successfully")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to initialize Git repository: {e}")
                raise
        elif not (self.repo_path / ".git").exists():
            logger.warning(f"Path {self.repo_path} exists but is not a Git repository")
            raise ValueError(f"Path {self.repo_path} is not a Git repository")
        else:
            logger.info(f"Using existing Git repository at {self.repo_path}")

    def _start_auto_commit_thread(self) -> None:
        """Start thread for automatic commits."""
        self._auto_commit_thread = threading.Thread(
            target=self._auto_commit_loop, daemon=True
        )
        self._auto_commit_thread.start()
        logger.info(f"Started auto-commit thread (interval: {self.commit_interval}s)")

    def _auto_commit_loop(self) -> None:
        """Background thread for automatic commits."""
        while not self._stop_event.is_set():
            time.sleep(self.commit_interval)
            if not self._stop_event.is_set():
                try:
                    if self._has_changes():
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.commit(f"Auto-commit at {timestamp}")
                        logger.debug(f"Auto-committed changes at {timestamp}")
                except Exception as e:
                    logger.error(f"Error in auto-commit: {e}")

    def _has_changes(self) -> bool:
        """Check if repository has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_path),
                check=True,
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check Git status: {e}")
            return False

    def commit(self, message: str) -> bool:
        """
        Commit changes to the repository.

        Args:
            message: Commit message

        Returns:
            True if commit was successful, False otherwise
        """
        try:
            # Add all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=str(self.repo_path),
                check=True,
                capture_output=True,
            )

            # Commit with message
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
            )

            # git commit returns exit code 1 if there's nothing to commit
            if (
                "nothing to commit" in result.stdout
                or "nothing to commit" in result.stderr
            ):
                logger.debug("No changes to commit")
                return False

            if result.returncode != 0:
                logger.warning(f"Git commit failed: {result.stderr}")
                return False

            logger.info(f"Committed changes: {message}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return False

    def save_memory(
        self, memory_type: str, memory_id: str, content: Dict[str, Any]
    ) -> bool:
        """
        Save memory content to the repository.

        Args:
            memory_type: Type of memory (e.g., "thoughts", "reflections")
            memory_id: Unique identifier for the memory
            content: Memory content to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            memory_dir = self.repo_path / memory_type
            os.makedirs(memory_dir, exist_ok=True)

            # Save content as JSON
            memory_path = memory_dir / f"{memory_id}.json"
            with open(memory_path, "w") as f:
                json.dump(content, f, indent=2)

            logger.debug(f"Saved {memory_type}/{memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save memory {memory_type}/{memory_id}: {e}")
            return False

    def load_memory(self, memory_type: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory content from the repository.

        Args:
            memory_type: Type of memory (e.g., "thoughts", "reflections")
            memory_id: Unique identifier for the memory

        Returns:
            Memory content or None if not found
        """
        memory_path = self.repo_path / memory_type / f"{memory_id}.json"
        if not memory_path.exists():
            logger.debug(f"Memory {memory_type}/{memory_id} not found")
            return None

        try:
            with open(memory_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory {memory_type}/{memory_id}: {e}")
            return None

    def list_memories(self, memory_type: str) -> List[str]:
        """
        List all memories of a specific type.

        Args:
            memory_type: Type of memory (e.g., "thoughts", "reflections")

        Returns:
            List of memory IDs
        """
        memory_dir = self.repo_path / memory_type
        if not memory_dir.exists():
            return []

        return [
            f.stem
            for f in memory_dir.glob("*.json")
            if f.is_file() and not f.name.startswith(".")
        ]

    def get_history(self, path: str, max_entries: int = 10) -> List[Dict[str, str]]:
        """
        Get commit history for a specific file or directory.

        Args:
            path: Relative path within the repository
            max_entries: Maximum number of history entries to return

        Returns:
            List of commit information (date, message, hash)
        """
        try:
            abs_path = self.repo_path / path
            rel_path = abs_path.relative_to(self.repo_path)

            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"-{max_entries}",
                    "--pretty=format:%H|%ad|%s",
                    "--date=iso",
                    "--",
                    str(rel_path),
                ],
                cwd=str(self.repo_path),
                check=True,
                capture_output=True,
                text=True,
            )

            if not result.stdout:
                return []

            history = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 2)
                if len(parts) != 3:
                    continue

                history.append(
                    {"hash": parts[0], "date": parts[1], "message": parts[2]}
                )

            return history

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get history for {path}: {e}")
            return []

    def close(self) -> None:
        """Clean up resources and stop auto-commit thread."""
        if self.auto_commit:
            self._stop_event.set()
            if hasattr(self, "_auto_commit_thread"):
                self._auto_commit_thread.join(timeout=1.0)

            # Final commit if there are changes
            if self._has_changes():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.commit(f"Final commit at {timestamp}")

        logger.info("Git memory manager closed")
