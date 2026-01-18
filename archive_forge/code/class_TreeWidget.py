from __future__ import annotations
import typing
from .columns import Columns
from .constants import WHSettings
from .listbox import ListBox, ListWalker
from .padding import Padding
from .text import Text
from .widget import WidgetWrap
from .wimp import SelectableIcon
class TreeWidget(WidgetWrap[Padding[typing.Union[Text, Columns]]]):
    """A widget representing something in a nested tree display."""
    indent_cols = 3
    unexpanded_icon = SelectableIcon('+', 0)
    expanded_icon = SelectableIcon('-', 0)

    def __init__(self, node: TreeNode) -> None:
        self._node = node
        self._innerwidget: Text | None = None
        self.is_leaf = not hasattr(node, 'get_first_child')
        self.expanded = True
        widget = self.get_indented_widget()
        super().__init__(widget)

    def selectable(self) -> bool:
        """
        Allow selection of non-leaf nodes so children may be (un)expanded
        """
        return not self.is_leaf

    def get_indented_widget(self) -> Padding[Text | Columns]:
        widget = self.get_inner_widget()
        if not self.is_leaf:
            widget = Columns([(1, [self.unexpanded_icon, self.expanded_icon][self.expanded]), widget], dividechars=1)
        indent_cols = self.get_indent_cols()
        return Padding(widget, width=(WHSettings.RELATIVE, 100), left=indent_cols)

    def update_expanded_icon(self) -> None:
        """Update display widget text for parent widgets"""
        icon = [self.unexpanded_icon, self.expanded_icon][self.expanded]
        self._w.base_widget.contents[0] = (icon, (WHSettings.GIVEN, 1, False))

    def get_indent_cols(self) -> int:
        return self.indent_cols * self.get_node().get_depth()

    def get_inner_widget(self) -> Text:
        if self._innerwidget is None:
            self._innerwidget = self.load_inner_widget()
        return self._innerwidget

    def load_inner_widget(self) -> Text:
        return Text(self.get_display_text())

    def get_node(self) -> TreeNode:
        return self._node

    def get_display_text(self) -> str | tuple[Hashable, str] | list[str | tuple[Hashable, str]]:
        return f'{self.get_node().get_key()}: {self.get_node().get_value()!s}'

    def next_inorder(self) -> TreeWidget | None:
        """Return the next TreeWidget depth first from this one."""
        first_child = self.first_child()
        if first_child is not None:
            return first_child
        this_node = self.get_node()
        next_node = this_node.next_sibling()
        depth = this_node.get_depth()
        while next_node is None and depth > 0:
            this_node = this_node.get_parent()
            next_node = this_node.next_sibling()
            depth -= 1
            if depth != this_node.get_depth():
                raise ValueError(depth)
        if next_node is None:
            return None
        return next_node.get_widget()

    def prev_inorder(self) -> TreeWidget | None:
        """Return the previous TreeWidget depth first from this one."""
        this_node = self._node
        prev_node = this_node.prev_sibling()
        if prev_node is not None:
            prev_widget = prev_node.get_widget()
            last_child = prev_widget.last_child()
            if last_child is None:
                return prev_widget
            return last_child
        depth = this_node.get_depth()
        if prev_node is None and depth == 0:
            return None
        if prev_node is None:
            prev_node = this_node.get_parent()
        return prev_node.get_widget()

    def keypress(self, size: tuple[int] | tuple[()], key: str) -> str | None:
        """Handle expand & collapse requests (non-leaf nodes)"""
        if self.is_leaf:
            return key
        if key in {'+', 'right'}:
            self.expanded = True
            self.update_expanded_icon()
            return None
        if key == '-':
            self.expanded = False
            self.update_expanded_icon()
            return None
        if self._w.selectable():
            return super().keypress(size, key)
        return key

    def mouse_event(self, size: tuple[int] | tuple[()], event: str, button: int, col: int, row: int, focus: bool) -> bool:
        if self.is_leaf or event != 'mouse press' or button != 1:
            return False
        if row == 0 and col == self.get_indent_cols():
            self.expanded = not self.expanded
            self.update_expanded_icon()
            return True
        return False

    def first_child(self) -> TreeWidget | None:
        """Return first child if expanded."""
        if self.is_leaf or not self.expanded:
            return None
        if self._node.has_children():
            first_node = self._node.get_first_child()
            return first_node.get_widget()
        return None

    def last_child(self) -> TreeWidget | None:
        """Return last child if expanded."""
        if self.is_leaf or not self.expanded:
            return None
        if self._node.has_children():
            last_child = self._node.get_last_child().get_widget()
        else:
            return None
        last_descendant = last_child.last_child()
        if last_descendant is None:
            return last_child
        return last_descendant