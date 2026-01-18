"""
Secure execution sandbox for Eidosian Forge.

Provides isolated environments for executing code with controlled access
to resources and timing constraints.
"""

import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Error raised when sandbox execution fails."""

    pass


class ExecutionTimeoutError(SandboxError):
    """Error raised when execution times out."""

    pass


class MemoryLimitError(SandboxError):
    """Error raised when memory limit is exceeded."""

    pass


class Sandbox:
    """
    Secure execution environment for running agent code.

    Provides isolation and resource limits to ensure that agent code
    cannot adversely affect the host system.
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        timeout_seconds: int = 60,
        max_memory_mb: int = 256,
        allow_network: bool = False,
    ):
        """
        Initialize sandbox environment.

        Args:
            workspace_dir: Directory for sandbox files (temporary if None)
            timeout_seconds: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            allow_network: Whether to allow network access
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.allow_network = allow_network

        # Set up workspace
        if workspace_dir:
            self.workspace = Path(workspace_dir)
            self.workspace.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="eidosian_sandbox_")
            self.workspace = Path(self._temp_dir.name)

        # Set up subdirectories
        self.bin_dir = self.workspace / "bin"
        self.data_dir = self.workspace / "data"
        self.tmp_dir = self.workspace / "tmp"

        self.bin_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

        logger.info(f"Sandbox initialized at {self.workspace}")

        # Track running processes for cleanup
        self._processes: List[subprocess.Popen] = []

    def execute_python(
        self, code: str, filename: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """
        Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            filename: Optional filename to save code to

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        if filename is None:
            # Generate a unique filename
            filename = f"script_{int(time.time())}_{hash(code) % 10000}.py"

        filepath = self.bin_dir / filename

        # Write code to file
        with open(filepath, "w") as f:
            f.write(code)

        return self._execute_python_file(filepath)

    def _execute_python_file(self, filepath: Path) -> Tuple[str, str, int]:
        """
        Execute a Python file in the sandbox.

        Args:
            filepath: Path to Python file

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        # Prepare environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workspace)

        if not self.allow_network:
            # Block network access with a fake socket module
            socket_blocker = self.bin_dir / "socket.py"
            with open(socket_blocker, "w") as f:
                f.write(
                    """
def error(*args, **kwargs):
    raise RuntimeError("Network access is disabled in the sandbox")

def warning(*args, **kwargs):
    print("WARNING: Network access attempt blocked")

# Block all socket functionality
def __getattr__(name):
    warning(f"Attempted to access socket.{name}")
    raise RuntimeError(f"Network access is disabled in the sandbox")
"""
                )

        # Create command
        cmd = [sys.executable, str(filepath)]

        # Execute with resource limits
        process = None
        stdout = ""
        stderr = ""
        return_code = -1

        try:
            # Create subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace),
                env=env,
                text=True,
                preexec_fn=self._set_process_limits,
            )
            self._processes.append(process)

            # Set timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise ExecutionTimeoutError(
                    f"Execution timed out after {self.timeout_seconds} seconds"
                )

        except OSError as e:
            stderr = f"Failed to execute process: {e}"

        finally:
            if process in self._processes:
                self._processes.remove(process)

        return stdout, stderr, return_code

    def _set_process_limits(self) -> None:
        """Set resource limits for child processes."""
        # Set memory limit
        memory_bytes = self.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # Prevent subprocess creation
        # resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

        # Handle signals
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

    def create_file(self, content: str, path: str) -> Path:
        """
        Create a file in the sandbox.

        Args:
            content: File content
            path: Relative path in workspace

        Returns:
            Path to created file
        """
        # Ensure path doesn't try to escape sandbox
        safe_path = self._sanitize_path(path)
        full_path = self.workspace / safe_path

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write content
        with open(full_path, "w") as f:
            f.write(content)

        logger.debug(f"Created file at {safe_path} in sandbox")
        return full_path

    def read_file(self, path: str) -> Optional[str]:
        """
        Read a file from the sandbox.

        Args:
            path: Relative path in workspace

        Returns:
            File content or None if file doesn't exist
        """
        # Ensure path doesn't try to escape sandbox
        safe_path = self._sanitize_path(path)
        full_path = self.workspace / safe_path

        if not full_path.exists():
            return None

        with open(full_path, "r") as f:
            return f.read()

    def _sanitize_path(self, path: str) -> str:
        """
        Sanitize a path to prevent directory traversal.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path
        """
        # Convert to Path and resolve to absolute path
        p = Path(path)

        # Remove leading dots and slashes
        parts = [part for part in p.parts if part != ".." and part != "."]
        safe_path = Path(*parts)

        return str(safe_path)

    def list_files(self, directory: str = "") -> List[str]:
        """
        List files in a sandbox directory.

        Args:
            directory: Relative directory path

        Returns:
            List of filenames
        """
        safe_dir = self._sanitize_path(directory)
        dir_path = self.workspace / safe_dir

        if not dir_path.exists() or not dir_path.is_dir():
            return []

        return [f.name for f in dir_path.iterdir()]

    def execute_command(self, command: List[str]) -> Tuple[str, str, int]:
        """
        Execute a shell command in the sandbox.

        Args:
            command: Command as list of arguments

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        allowed_commands = {
            "ls",
            "cat",
            "head",
            "tail",
            "grep",
            "find",
            "wc",
            "sort",
            "uniq",
            "echo",
            "pwd",
        }

        # Safety check - only allow specific commands
        if command[0] not in allowed_commands:
            return "", f"Command '{command[0]}' not allowed in sandbox", 1

        # Execute with resource limits
        process = None
        stdout = ""
        stderr = ""
        return_code = -1

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.workspace),
                text=True,
                preexec_fn=self._set_process_limits,
            )
            self._processes.append(process)

            # Set timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise ExecutionTimeoutError(
                    f"Execution timed out after {self.timeout_seconds} seconds"
                )

        except OSError as e:
            stderr = f"Failed to execute command: {e}"

        finally:
            if process in self._processes:
                self._processes.remove(process)

        return stdout, stderr, return_code

    def close(self) -> None:
        """Clean up resources."""
        # Terminate any running processes
        for process in self._processes[
            :
        ]:  # Make a copy since we'll modify during iteration
            try:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                self._processes.remove(process)
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")

        # Clean up temporary directory if we created one
        if self._temp_dir:
            self._temp_dir.cleanup()

        logger.info("Sandbox closed and resources cleaned up")


# Convenience function for executing code with a temporary sandbox
def run_in_sandbox(
    code: str, timeout_seconds: int = 30, max_memory_mb: int = 128
) -> Tuple[str, str, int]:
    """
    Run code in a temporary sandbox.

    Args:
        code: Python code to execute
        timeout_seconds: Maximum execution time
        max_memory_mb: Maximum memory usage in MB

    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    sandbox = Sandbox(timeout_seconds=timeout_seconds, max_memory_mb=max_memory_mb)

    try:
        return sandbox.execute_python(code)
    finally:
        sandbox.close()
