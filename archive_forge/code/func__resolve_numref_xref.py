import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
def _resolve_numref_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
    if target in self.labels:
        docname, labelid, figname = self.labels.get(target, ('', '', ''))
    else:
        docname, labelid = self.anonlabels.get(target, ('', ''))
        figname = None
    if not docname:
        return None
    target_node = env.get_doctree(docname).ids.get(labelid)
    figtype = self.get_enumerable_node_type(target_node)
    if figtype is None:
        return None
    if figtype != 'section' and env.config.numfig is False:
        logger.warning(__('numfig is disabled. :numref: is ignored.'), location=node)
        return contnode
    try:
        fignumber = self.get_fignumber(env, builder, figtype, docname, target_node)
        if fignumber is None:
            return contnode
    except ValueError:
        logger.warning(__('Failed to create a cross reference. Any number is not assigned: %s'), labelid, location=node)
        return contnode
    try:
        if node['refexplicit']:
            title = contnode.astext()
        else:
            title = env.config.numfig_format.get(figtype, '')
        if figname is None and '{name}' in title:
            logger.warning(__('the link has no caption: %s'), title, location=node)
            return contnode
        else:
            fignum = '.'.join(map(str, fignumber))
            if '{name}' in title or 'number' in title:
                if figname:
                    newtitle = title.format(name=figname, number=fignum)
                else:
                    newtitle = title.format(number=fignum)
            else:
                newtitle = title % fignum
    except KeyError as exc:
        logger.warning(__('invalid numfig_format: %s (%r)'), title, exc, location=node)
        return contnode
    except TypeError:
        logger.warning(__('invalid numfig_format: %s'), title, location=node)
        return contnode
    return self.build_reference_node(fromdocname, builder, docname, labelid, newtitle, 'numref', nodeclass=addnodes.number_reference, title=title)