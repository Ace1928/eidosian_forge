from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException
class DisorderedEntityWrapper:
    """Wrapper class to group equivalent Entities.

    This class is a simple wrapper class that groups a number of equivalent
    Entities and forwards all method calls to one of them (the currently selected
    object). DisorderedResidue and DisorderedAtom are subclasses of this class.

    E.g.: A DisorderedAtom object contains a number of Atom objects,
    where each Atom object represents a specific position of a disordered
    atom in the structure.
    """

    def __init__(self, id):
        """Initialize the class."""
        self.id = id
        self.child_dict = {}
        self.selected_child = None
        self.parent = None

    def __getattr__(self, method):
        """Forward the method call to the selected child."""
        if method == '__setstate__':
            raise AttributeError
        if not hasattr(self, 'selected_child'):
            raise AttributeError
        return getattr(self.selected_child, method)

    def __getitem__(self, id):
        """Return the child with the given id."""
        return self.selected_child[id]

    def __setitem__(self, id, child):
        """Add a child, associated with a certain id."""
        self.child_dict[id] = child

    def __contains__(self, id):
        """Check if the child has the given id."""
        return id in self.selected_child

    def __iter__(self):
        """Return the number of children."""
        return iter(self.selected_child)

    def __len__(self):
        """Return the number of children."""
        return len(self.selected_child)

    def __sub__(self, other):
        """Subtraction with another object."""
        return self.selected_child - other

    def __gt__(self, other):
        """Return if child is greater than other."""
        return self.selected_child > other

    def __ge__(self, other):
        """Return if child is greater or equal than other."""
        return self.selected_child >= other

    def __lt__(self, other):
        """Return if child is less than other."""
        return self.selected_child < other

    def __le__(self, other):
        """Return if child is less or equal than other."""
        return self.selected_child <= other

    def copy(self):
        """Copy disorderd entity recursively."""
        shallow = copy(self)
        shallow.child_dict = {}
        shallow.detach_parent()
        for child in self.disordered_get_list():
            shallow.disordered_add(child.copy())
        return shallow

    def get_id(self):
        """Return the id."""
        return self.id

    def disordered_has_id(self, id):
        """Check if there is an object present associated with this id."""
        return id in self.child_dict

    def detach_parent(self):
        """Detach the parent."""
        self.parent = None
        for child in self.disordered_get_list():
            child.detach_parent()

    def get_parent(self):
        """Return parent."""
        return self.parent

    def set_parent(self, parent):
        """Set the parent for the object and its children."""
        self.parent = parent
        for child in self.disordered_get_list():
            child.set_parent(parent)

    def disordered_select(self, id):
        """Select the object with given id as the currently active object.

        Uncaught method calls are forwarded to the selected child object.
        """
        self.selected_child = self.child_dict[id]

    def disordered_add(self, child):
        """Add disordered entry.

        This is implemented by DisorderedAtom and DisorderedResidue.
        """
        raise NotImplementedError

    def disordered_remove(self, child):
        """Remove disordered entry.

        This is implemented by DisorderedAtom and DisorderedResidue.
        """
        raise NotImplementedError

    def is_disordered(self):
        """Return 2, indicating that this Entity is a collection of Entities."""
        return 2

    def disordered_get_id_list(self):
        """Return a list of id's."""
        return sorted(self.child_dict)

    def disordered_get(self, id=None):
        """Get the child object associated with id.

        If id is None, the currently selected child is returned.
        """
        if id is None:
            return self.selected_child
        return self.child_dict[id]

    def disordered_get_list(self):
        """Return list of children."""
        return list(self.child_dict.values())