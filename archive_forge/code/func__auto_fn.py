from __future__ import annotations
from typing import Callable
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from .. import util
def _auto_fn(name: str) -> Optional[Callable[[], Type[Dialect]]]:
    """default dialect importer.

    plugs into the :class:`.PluginLoader`
    as a first-hit system.

    """
    if '.' in name:
        dialect, driver = name.split('.')
    else:
        dialect = name
        driver = 'base'
    try:
        if dialect == 'mariadb':
            module = __import__('sqlalchemy.dialects.mysql.mariadb').dialects.mysql.mariadb
            return module.loader(driver)
        else:
            module = __import__('sqlalchemy.dialects.%s' % (dialect,)).dialects
            module = getattr(module, dialect)
    except ImportError:
        return None
    if hasattr(module, driver):
        module = getattr(module, driver)
        return lambda: module.dialect
    else:
        return None