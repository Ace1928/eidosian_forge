import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def _ensure_cache_dir(*, target_dir):
    """Prepare local cache directory if it doesn't exist already.

    Creates::

        /path/to/target_dir/
                 └─ data/
                    └─ README.txt
    """
    os.makedirs(osp.join(target_dir, 'data'), exist_ok=True)
    readme_src = osp.join(_DISTRIBUTION_DIR, 'data/README.txt')
    readme_dest = osp.join(target_dir, 'data/README.txt')
    if not osp.exists(readme_dest):
        shutil.copy2(readme_src, readme_dest)