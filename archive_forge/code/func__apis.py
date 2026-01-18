from __future__ import annotations
from sphinx.util import logging  # isort:skip
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import EXAMPLE_METADATA
from .util import get_sphinx_resources
def _apis(apis: str | None) -> str | None:
    if apis is None:
        return
    apis = apis.split('#')[0].strip()
    results = []
    for api in (api.strip() for api in apis.split(',')):
        last = api.split('.')[-1]
        if api.startswith('bokeh.models'):
            results.append(f':class:`bokeh.models.{last} <{api}>`')
        elif 'figure.' in api:
            results.append(f':meth:`figure.{last} <{api}>`')
        elif 'GMap.' in api:
            results.append(f':meth:`GMap.{last} <{api}>`')
        else:
            results.append(f':class:`{api}`')
    return ', '.join(results)