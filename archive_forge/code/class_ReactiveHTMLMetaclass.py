from __future__ import annotations
import datetime as dt
import difflib
import inspect
import logging
import re
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache, partial
from pprint import pformat
from typing import (
import numpy as np
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.model import DataModel
from bokeh.models import ImportedStyleSheet
from packaging.version import Version
from param.parameterized import (
from .io.document import unlocked
from .io.model import hold
from .io.notebook import push
from .io.resources import (
from .io.state import set_curdoc, state
from .models.reactive_html import (
from .util import (
from .viewable import Layoutable, Renderable, Viewable
class ReactiveHTMLMetaclass(ParameterizedMetaclass):
    """
    Parses the ReactiveHTML._template of the class and initializes
    variables, callbacks and the data model to sync the parameters and
    HTML attributes.
    """
    _loaded_extensions: ClassVar[Set[str]] = set()
    _name_counter: ClassVar[Counter] = Counter()
    _script_regex: ClassVar[str] = 'script\\([\\"|\'](.*)[\\"|\']\\)'

    def __init__(mcs, name: str, bases: Tuple[Type, ...], dict_: Mapping[str, Any]):
        from .io.datamodel import PARAM_MAPPING, construct_data_model
        mcs.__original_doc__ = mcs.__doc__
        ParameterizedMetaclass.__init__(mcs, name, bases, dict_)
        cls_name = mcs.__name__
        for name, child_type in mcs._child_config.items():
            if name not in mcs.param:
                raise ValueError(f'{cls_name}._child_config for {name!r} does not match any parameters. Ensure the name of each child config matches one of the parameters.')
            elif child_type not in ('model', 'template', 'literal'):
                raise ValueError(f"{cls_name}._child_config for {name!r} child parameter declares unknown type {{child_type!r}}. The '_child_config' mode must be one of 'model', 'template' or 'literal'.")
        mcs._parser = ReactiveHTMLParser(mcs)
        mcs._parser.feed(mcs._template)
        if mcs._parser._open_for:
            raise ValueError(f'{cls_name}._template contains for loop without closing {{% endfor %}} statement.')
        if mcs._parser._node_stack:
            raise ValueError(f'{cls_name}._template contains tags which were never closed. Ensure all tags in your template have a matching closing tag, e.g. if there is a tag <div>, ensure there is a matching </div> tag.')
        mcs._node_callbacks: Dict[str, List[Tuple[str, str]]] = {}
        mcs._inline_callbacks = []
        for node, attrs in mcs._parser.attrs.items():
            for attr, parameters, _template in attrs:
                for p in parameters:
                    if p in mcs.param or '.' in p:
                        continue
                    if re.match(mcs._script_regex, p):
                        name = re.findall(mcs._script_regex, p)[0]
                        if name not in mcs._scripts:
                            raise ValueError(f'{cls_name}._template inline callback references unknown script {name!r}, ensure the referenced script is declaredin the _scripts dictionary.')
                        if node not in mcs._node_callbacks:
                            mcs._node_callbacks[node] = []
                        mcs._node_callbacks[node].append((attr, p))
                    elif hasattr(mcs, p):
                        if node not in mcs._node_callbacks:
                            mcs._node_callbacks[node] = []
                        mcs._node_callbacks[node].append((attr, p))
                        mcs._inline_callbacks.append((node, attr, p))
                    else:
                        matches = difflib.get_close_matches(p, dir(mcs))
                        raise ValueError(f"{cls_name}._template references unknown parameter or method '{p}', similar parameters and methods include {matches}.")
        ignored = list(Reactive.param)
        types = {}
        for child in mcs._parser.children.values():
            cparam = mcs.param[child]
            if mcs._child_config.get(child) == 'literal':
                types[child] = param.String
            elif type(cparam) not in PARAM_MAPPING or isinstance(cparam, (param.List, param.Dict, param.Tuple)) or (isinstance(cparam, param.ClassSelector) and isinstance(cparam.class_, type) and (not issubclass(cparam.class_, param.Parameterized) or issubclass(cparam.class_, Reactive))):
                ignored.append(child)
        ignored.remove('name')
        ReactiveHTMLMetaclass._name_counter[name] += 1
        model_name = f'{name}{ReactiveHTMLMetaclass._name_counter[name]}'
        mcs._data_model = construct_data_model(mcs, name=model_name, ignore=ignored, types=types)