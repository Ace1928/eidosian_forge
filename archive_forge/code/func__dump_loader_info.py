from __future__ import annotations
import typing as t
from jinja2.loaders import BaseLoader
from werkzeug.routing import RequestRedirect
from .blueprints import Blueprint
from .globals import request_ctx
from .sansio.app import App
def _dump_loader_info(loader: BaseLoader) -> t.Iterator[str]:
    yield f'class: {type(loader).__module__}.{type(loader).__name__}'
    for key, value in sorted(loader.__dict__.items()):
        if key.startswith('_'):
            continue
        if isinstance(value, (tuple, list)):
            if not all((isinstance(x, str) for x in value)):
                continue
            yield f'{key}:'
            for item in value:
                yield f'  - {item}'
            continue
        elif not isinstance(value, (str, int, float, bool)):
            continue
        yield f'{key}: {value!r}'