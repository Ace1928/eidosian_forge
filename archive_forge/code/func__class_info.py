import builtins
import inspect
import re
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.graphviz import (figure_wrapper, graphviz, render_dot_html, render_dot_latex,
from sphinx.util import md5
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.texinfo import TexinfoTranslator
def _class_info(self, classes: List[Any], show_builtins: bool, private_bases: bool, parts: int, aliases: Dict[str, str], top_classes: List[Any]) -> List[Tuple[str, str, List[str], str]]:
    """Return name and bases for all classes that are ancestors of
        *classes*.

        *parts* gives the number of dotted name parts to include in the
        displayed node names, from right to left. If given as a negative, the
        number of parts to drop from the left. A value of 0 displays the full
        dotted name. E.g. ``sphinx.ext.inheritance_diagram.InheritanceGraph``
        with ``parts=2`` or ``parts=-2`` gets displayed as
        ``inheritance_diagram.InheritanceGraph``, and as
        ``ext.inheritance_diagram.InheritanceGraph`` with ``parts=3`` or
        ``parts=-1``.

        *top_classes* gives the name(s) of the top most ancestor class to
        traverse to. Multiple names can be specified separated by comma.
        """
    all_classes = {}

    def recurse(cls: Any) -> None:
        if not show_builtins and cls in py_builtins:
            return
        if not private_bases and cls.__name__.startswith('_'):
            return
        nodename = self.class_name(cls, parts, aliases)
        fullname = self.class_name(cls, 0, aliases)
        tooltip = None
        try:
            if cls.__doc__:
                doc = cls.__doc__.strip().split('\n')[0]
                if doc:
                    tooltip = '"%s"' % doc.replace('"', '\\"')
        except Exception:
            pass
        baselist: List[str] = []
        all_classes[cls] = (nodename, fullname, baselist, tooltip)
        if fullname in top_classes:
            return
        for base in cls.__bases__:
            if not show_builtins and base in py_builtins:
                continue
            if not private_bases and base.__name__.startswith('_'):
                continue
            baselist.append(self.class_name(base, parts, aliases))
            if base not in all_classes:
                recurse(base)
    for cls in classes:
        recurse(cls)
    return list(all_classes.values())