import warnings
from pathlib import Path
from typing import Optional, Type, Union
from lightning_fabric.utilities.rank_zero import LightningDeprecationWarning
def _is_path_in_lightning(path: Path) -> bool:
    """Naive check whether the path looks like a path from the lightning package."""
    return 'lightning' in str(path.absolute())