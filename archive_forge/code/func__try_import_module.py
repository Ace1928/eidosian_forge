import functools
import sys
from lightning_utilities.core.imports import RequirementCache, package_available
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@functools.lru_cache(maxsize=128)
def _try_import_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except (ImportError, AttributeError) as err:
        rank_zero_warn(f'Import of {module_name} package failed for some compatibility issues:\n{err}')
        return False