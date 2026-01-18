import logging
from typing import Optional, Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import (
@classmethod
def _resolve_filename(cls, root_log_dir: Path, filename: str) -> Path:
    """
        Resolves the file path relative to the root log directory.

        Args:
            root_log_dir: Root log directory.
            filename: File path relative to the root log directory.

        Raises:
            FileNotFoundError: If the file path is invalid.

        Returns:
            The absolute file path resolved from the root log directory.
        """
    if not Path(filename).is_absolute():
        filepath = root_log_dir / filename
    else:
        filepath = Path(filename)
    filepath = Path(os.path.abspath(filepath))
    if not filepath.is_file():
        raise FileNotFoundError(f'A file is not found at: {filepath}')
    try:
        filepath.relative_to(root_log_dir)
    except ValueError as e:
        raise FileNotFoundError(f'{filepath} not in {root_log_dir}: {e}')
    return filepath.resolve()