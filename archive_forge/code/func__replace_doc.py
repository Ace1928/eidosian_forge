import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat
def _replace_doc(source_obj: object, target_obj: object, overwrite: bool, apilink: Optional[Union[str, List[str]]], parent_cls: Optional[Fn]=None, attr_name: Optional[str]=None) -> None:
    """
    Replace docstring in `target_obj`, possibly taking from `source_obj` and augmenting.

    Can append the link to pandas API online documentation.

    Parameters
    ----------
    source_obj : object
        Any object from which to take docstring from.
    target_obj : object
        The object which docstring to replace.
    overwrite : bool
        Forces replacing the docstring with the one from `source_obj` even
        if `target_obj` has its own non-empty docstring.
    apilink : str | List[str], optional
        If non-empty, insert the link(s) to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".
    parent_cls : class, optional
        If `target_obj` is an attribute of a class, `parent_cls` should be that class.
        This is used for generating the API URL as well as for handling special cases
        like `target_obj` being a property.
    attr_name : str, optional
        Gives the name to `target_obj` if it's an attribute of `parent_cls`.
        Needed to handle some special cases and in most cases could be determined automatically.
    """
    if isinstance(target_obj, (staticmethod, classmethod)):
        target_obj = target_obj.__func__
    source_doc = source_obj.__doc__ or ''
    target_doc = target_obj.__doc__ or ''
    overwrite = overwrite or not target_doc
    doc = source_doc if overwrite else target_doc
    if parent_cls and (not attr_name):
        if isinstance(target_obj, property):
            attr_name = target_obj.fget.__name__
        elif isinstance(target_obj, (staticmethod, classmethod)):
            attr_name = target_obj.__func__.__name__
        else:
            attr_name = target_obj.__name__
    if source_doc.strip() and apilink and ('pandas API documentation for ' not in target_doc) and (not (attr_name or '').startswith('_')):
        apilink_l = [apilink] if not isinstance(apilink, list) and apilink else apilink
        links = []
        for link in apilink_l:
            if attr_name:
                token = f'{link}.{attr_name}'
            else:
                token = link
            url = _make_api_url(token)
            links.append(f'`{token} <{url}>`_')
        indent_line = ' ' * _get_indent(doc)
        notes_section = f'\n{indent_line}Notes\n{indent_line}-----\n'
        url_line = f'{indent_line}See pandas API documentation for {', '.join(links)} for more.\n'
        notes_section_with_url = notes_section + url_line
        if notes_section in doc:
            doc = doc.replace(notes_section, notes_section_with_url)
        else:
            doc += notes_section_with_url
    if parent_cls and isinstance(target_obj, property):
        if overwrite:
            target_obj.fget.__doc_inherited__ = True
        assert attr_name is not None
        setattr(parent_cls, attr_name, property(target_obj.fget, target_obj.fset, target_obj.fdel, doc))
    else:
        if overwrite:
            target_obj.__doc_inherited__ = True
        target_obj.__doc__ = doc