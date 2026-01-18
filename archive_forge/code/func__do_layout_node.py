from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def _do_layout_node(self, node, level, y):
    if self.hide_root and node is self.root:
        level -= 1
    else:
        node.x = self.x + self.indent_start + level * self.indent_level
        node.top = y
        if node.size_hint_x:
            node.width = (self.width - (node.x - self.x)) * node.size_hint_x
        y -= node.height
        if not node.is_open:
            return y
    for cnode in node.nodes:
        y = self._do_layout_node(cnode, level + 1, y)
    return y