import re
from typing import TYPE_CHECKING, Any, Dict, List, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.parsers.rst.directives.misc import Class
from docutils.parsers.rst.directives.misc import Include as BaseInclude
from sphinx import addnodes
from sphinx.domains.changeset import VersionChange  # NOQA  # for compatibility
from sphinx.locale import _, __
from sphinx.util import docname_join, logging, url_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.matching import Matcher, patfilter
from sphinx.util.nodes import explicit_title_re
from sphinx.util.typing import OptionSpec
class HList(SphinxDirective):
    """
    Directive for a list that gets compacted horizontally.
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: OptionSpec = {'columns': int}

    def run(self) -> List[Node]:
        ncolumns = self.options.get('columns', 2)
        node = nodes.paragraph()
        node.document = self.state.document
        self.state.nested_parse(self.content, self.content_offset, node)
        if len(node.children) != 1 or not isinstance(node.children[0], nodes.bullet_list):
            logger.warning(__('.. hlist content is not a list'), location=(self.env.docname, self.lineno))
            return []
        fulllist = node.children[0]
        npercol, nmore = divmod(len(fulllist), ncolumns)
        index = 0
        newnode = addnodes.hlist()
        newnode['ncolumns'] = str(ncolumns)
        for column in range(ncolumns):
            endindex = index + (npercol + 1 if column < nmore else npercol)
            bullet_list = nodes.bullet_list()
            bullet_list += fulllist.children[index:endindex]
            newnode += addnodes.hlistcol('', bullet_list)
            index = endindex
        return [newnode]