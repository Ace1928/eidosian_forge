import copy
import weakref
from pyomo.common.autoslots import AutoSlots
class ICategorizedObjectContainer(ICategorizedObject):
    """
    Interface for categorized containers of categorized
    objects.
    """
    _is_container = True
    _child_storage_delimiter_string = None
    _child_storage_entry_string = None
    __slots__ = ()

    def activate(self, shallow=True):
        """Activate this container."""
        super(ICategorizedObjectContainer, self).activate()
        if not shallow:
            for child in self.children():
                if not child._is_container:
                    child.activate()
                else:
                    child.activate(shallow=False)

    def deactivate(self, shallow=True):
        """Deactivate this container."""
        super(ICategorizedObjectContainer, self).deactivate()
        if not shallow:
            for child in self.children():
                if not child._is_container:
                    child.deactivate()
                else:
                    child.deactivate(shallow=False)

    def child(self, *args, **kwds):
        """Returns a child of this container given a storage
        key."""
        raise NotImplementedError

    def children(self, *args, **kwds):
        """A generator over the children of this container."""
        raise NotImplementedError

    def components(self, *args, **kwds):
        """A generator over the set of components stored
        under this container."""
        raise NotImplementedError