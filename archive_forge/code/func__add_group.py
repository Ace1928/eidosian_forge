import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def _add_group(self, group):
    self.group_map[group] = {}
    if group.parent is None:
        self.top_groups.append(group)
    else:
        if group.parent not in self.group_map:
            self._add_group(group.parent)
        if group.parent not in self.group_children:
            self.group_children[group.parent] = []
        self.group_children[group.parent].append(group)
    group._assigned_batches.add(self)
    self._draw_list_dirty = True