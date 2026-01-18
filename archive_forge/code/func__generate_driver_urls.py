from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
def _generate_driver_urls(url, extra_drivers):
    main_driver = url.get_driver_name()
    extra_drivers.discard(main_driver)
    url = generate_driver_url(url, main_driver, '')
    yield url
    for drv in list(extra_drivers):
        if '?' in drv:
            driver_only, query_str = drv.split('?', 1)
        else:
            driver_only = drv
            query_str = None
        new_url = generate_driver_url(url, driver_only, query_str)
        if new_url:
            extra_drivers.remove(drv)
            yield new_url