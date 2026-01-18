import os
from pathlib import Path
from typing import List, Optional
def _paths_from_env(variable: str, default: List[Path]) -> List[Path]:
    """Read an environment variable as a list of paths.

    The environment variable with the specified name is read, and its
    value split on colons and returned as a list of paths. If the
    environment variable is not set, or set to the empty string, the
    default value is returned. Relative paths are ignored, as per the
    specification.

    Parameters
    ----------
    variable : str
        Name of the environment variable.
    default : List[Path]
        Default value.

    Returns
    -------
    List[Path]
        Value from environment or default.

    """
    value = os.environ.get(variable)
    if value:
        paths = [Path(path) for path in value.split(':') if os.path.isabs(path)]
        if paths:
            return paths
    return default