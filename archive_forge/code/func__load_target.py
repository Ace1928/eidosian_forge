import os.path
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .vision import VisionDataset
def _load_target(self, id: int) -> List[str]:
    return [ann['caption'] for ann in super()._load_target(id)]