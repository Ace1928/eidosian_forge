import pkgutil
from typing import Optional
from absl import logging
def _fetch_discovery_doc_from_pkg(package: str, resource: str) -> Optional[bytes]:
    """Loads a discovery doc as `bytes` specified by `package` and `resource` returning None on error."""
    try:
        raw_doc = pkgutil.get_data(package, resource)
    except ImportError:
        raw_doc = None
    if not raw_doc:
        logging.warning('Failed to load discovery doc from (package, resource): %s, %s', package, resource)
    else:
        logging.info('Successfully loaded discovery doc from (package, resource): %s, %s', package, resource)
    return raw_doc