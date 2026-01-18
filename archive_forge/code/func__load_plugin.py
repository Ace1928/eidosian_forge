import logging
import sys
from typing import Any, List
from .threading import run_once
from importlib.metadata import EntryPoint, entry_points
def _load_plugin(entrypoint: str, plugin: EntryPoint) -> None:
    logger = logging.getLogger(_load_plugin.__name__)
    try:
        res = plugin.load()
        if callable(res):
            res()
        logger.debug('loaded %s %s', entrypoint, plugin)
    except Exception as e:
        logger.debug('failed to load %s %s: %s', entrypoint, plugin, e)