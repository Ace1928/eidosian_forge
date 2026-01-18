import copy
import weakref
from pyomo.common.autoslots import AutoSlots
class ICategorizedObject(AutoSlots.Mixin):
    """
    Interface for objects that maintain a weak reference to
    a parent storage object and have a category type.

    This class is abstract. It assumes any derived class
    declares the attributes below with or without slots:

    Attributes:
        _ctype: Stores the object's category type, which
            should be some class derived from
            ICategorizedObject. This attribute may be
            declared at the class level.
        _parent: Stores a weak reference to the object's
            parent container or :const:`None`.
        _storage_key: Stores key this object can be accessed
            with through its parent container.
        _active (bool): Stores the active status of this
            object.
    """
    __slots__ = ()
    __autoslot_mappers__ = {'_parent': AutoSlots.weakref_mapper}
    _is_container = False
    'A flag used to indicate that the class is an instance\n    of ICategorizedObjectContainer.'
    _is_heterogeneous_container = False
    'A flag used to indicate that the class is an instance\n    of ICategorizedObjectContainer that stores objects with\n    different category types than its own.'

    @property
    def ctype(self):
        """The object's category type."""
        return self._ctype

    @property
    def parent(self):
        """The object's parent (possibly None)."""
        if isinstance(self._parent, weakref.ReferenceType):
            return self._parent()
        else:
            return self._parent

    @property
    def storage_key(self):
        """The object's storage key within its parent"""
        return self._storage_key

    @property
    def active(self):
        """The active status of this object."""
        return self._active

    @active.setter
    def active(self, value):
        raise AttributeError('Assignment not allowed. Use the (de)activate method')

    def _update_parent_and_storage_key(self, parent, key):
        object.__setattr__(self, '_parent', weakref.ref(parent))
        object.__setattr__(self, '_storage_key', key)

    def _clear_parent_and_storage_key(self):
        object.__setattr__(self, '_parent', None)
        object.__setattr__(self, '_storage_key', None)

    def activate(self):
        """Activate this object."""
        object.__setattr__(self, '_active', True)

    def deactivate(self):
        """Deactivate this object."""
        object.__setattr__(self, '_active', False)

    def getname(self, fully_qualified=False, name_buffer={}, convert=str, relative_to=None):
        """
        Dynamically generates a name for this object.

        Args:
            fully_qualified (bool): Generate a full name by
                iterating through all anscestor containers.
                Default is :const:`False`.
            convert (function): A function that converts a
                storage key into a string
                representation. Default is the built-in
                function str.
            relative_to (object): When generating a fully
                qualified name, generate the name relative
                to this block.

        Returns:
            If a parent exists, this method returns a string
            representing the name of the object in the
            context of its parent; otherwise (if no parent
            exists), this method returns :const:`None`.
        """
        assert fully_qualified or relative_to is None
        parent = self.parent
        if parent is None:
            return None
        key = self.storage_key
        name = parent._child_storage_entry_string % convert(key)
        if fully_qualified:
            parent_name = parent.getname(fully_qualified=True, relative_to=relative_to)
            if parent_name is not None and (relative_to is None or parent is not relative_to):
                return parent_name + parent._child_storage_delimiter_string + name
            else:
                return name
        else:
            return name

    @property
    def name(self):
        """The object's fully qualified name. Alias for
        `obj.getname(fully_qualified=True)`."""
        return self.getname(fully_qualified=True)

    @property
    def local_name(self):
        """The object's local name within the context of its
        parent. Alias for
        `obj.getname(fully_qualified=False)`."""
        return self.getname(fully_qualified=False)

    def __str__(self):
        """Convert this object to a string by first
        attempting to generate its fully qualified name. If
        the object does not have a name (because it does not
        have a parent, then a string containing the class
        name is returned."""
        name = self.name
        if name is None:
            return '<' + self.__class__.__name__ + '>'
        else:
            return name

    def clone(self):
        """
        Returns a copy of this object with the parent
        pointer set to :const:`None`.

        A clone is almost equivalent to deepcopy except that
        any categorized objects encountered that are not
        descendents of this object will reference the same
        object on the clone.
        """
        save_parent = self._parent
        object.__setattr__(self, '_parent', None)
        try:
            new_block = copy.deepcopy(self, {'__block_scope__': {id(self): True, id(None): False}})
        finally:
            object.__setattr__(self, '_parent', save_parent)
        return new_block

    def __deepcopy__(self, memo):
        if '__block_scope__' in memo:
            _known = memo['__block_scope__']
            _new = None
            tmp = self.parent
            _in_scope = tmp is None
            while id(tmp) not in _known:
                _new = (_new, id(tmp))
                tmp = tmp.parent
            _in_scope |= _known[id(tmp)]
            while _new is not None:
                _new, _id = _new
                _known[_id] = _in_scope
            if not _in_scope and id(self) not in _known:
                memo[id(self)] = self
                return self
        if id(self) in memo:
            return memo[id(self)]
        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans