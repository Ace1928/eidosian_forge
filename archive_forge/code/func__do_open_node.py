from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def _do_open_node(self, node):
    if self.hide_root and node is self.root:
        height = 0
    else:
        self.add_widget(node)
        height = node.height
        if not node.is_open:
            return height
    for cnode in node.nodes:
        height += self._do_open_node(cnode)
    return height