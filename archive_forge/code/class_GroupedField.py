from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.typing import TextlikeNode
class GroupedField(Field):
    """
    A doc field that is grouped; i.e., all fields of that type will be
    transformed into one field with its body being a bulleted list.  It always
    has an argument.  The argument can be linked using the given *rolename*.
    GroupedField should be used for doc fields that can occur more than once.
    If *can_collapse* is true, this field will revert to a Field if only used
    once.

    Example::

       :raises ErrorClass: description when it is raised
    """
    is_grouped = True
    list_type = nodes.bullet_list

    def __init__(self, name: str, names: Tuple[str, ...]=(), label: str=None, rolename: str=None, can_collapse: bool=False) -> None:
        super().__init__(name, names, label, True, rolename)
        self.can_collapse = can_collapse

    def make_field(self, types: Dict[str, List[Node]], domain: str, items: Tuple, env: BuildEnvironment=None, inliner: Inliner=None, location: Node=None) -> nodes.field:
        fieldname = nodes.field_name('', self.label)
        listnode = self.list_type()
        for fieldarg, content in items:
            par = nodes.paragraph()
            par.extend(self.make_xrefs(self.rolename, domain, fieldarg, addnodes.literal_strong, env=env, inliner=inliner, location=location))
            par += nodes.Text(' -- ')
            par += content
            listnode += nodes.list_item('', par)
        if len(items) == 1 and self.can_collapse:
            list_item = cast(nodes.list_item, listnode[0])
            fieldbody = nodes.field_body('', list_item[0])
            return nodes.field('', fieldname, fieldbody)
        fieldbody = nodes.field_body('', listnode)
        return nodes.field('', fieldname, fieldbody)