from __future__ import annotations
from collections import defaultdict, namedtuple
from typing import (
import param
from bokeh.models import Row as BkRow
from param.parameterized import iscoroutinefunction, resolve_ref
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import Column as PnColumn
from ..reactive import Reactive
from ..util import param_name, param_reprs, param_watchers
class ListLike(param.Parameterized):
    objects = param.List(default=[], doc='\n        The list of child objects that make up the layout.')
    _preprocess_params: ClassVar[List[str]] = ['objects']

    def __getitem__(self, index: int | slice) -> Viewable | List[Viewable]:
        return self.objects[index]

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[Viewable]:
        for obj in self.objects:
            yield obj

    def __iadd__(self, other: Iterable[Any]) -> 'ListLike':
        self.extend(other)
        return self

    def __add__(self, other: Iterable[Any]) -> 'ListLike':
        if isinstance(other, ListLike):
            other = other.objects
        else:
            other = list(other)
        return self.clone(*self.objects + other)

    def __radd__(self, other: Iterable[Any]) -> 'ListLike':
        if isinstance(other, ListLike):
            other = other.objects
        else:
            other = list(other)
        return self.clone(*other + self.objects)

    def __contains__(self, obj: Viewable) -> bool:
        return obj in self.objects

    def __setitem__(self, index: int | slice, panes: Iterable[Any]) -> None:
        from ..pane import panel
        new_objects = list(self)
        if not isinstance(index, slice):
            start, end = (index, index + 1)
            if start > len(self.objects):
                raise IndexError('Index %d out of bounds on %s containing %d objects.' % (end, type(self).__name__, len(self.objects)))
            panes = [panes]
        else:
            start = index.start or 0
            end = len(self) if index.stop is None else index.stop
            if index.start is None and index.stop is None:
                if not isinstance(panes, list):
                    raise IndexError('Expected a list of objects to replace the objects in the %s, got a %s type.' % (type(self).__name__, type(panes).__name__))
                expected = len(panes)
                new_objects = [None] * expected
                end = expected
            elif end > len(self.objects):
                raise IndexError('Index %d out of bounds on %s containing %d objects.' % (end, type(self).__name__, len(self.objects)))
            else:
                expected = end - start
            if not isinstance(panes, list) or len(panes) != expected:
                raise IndexError('Expected a list of %d objects to set on the %s to match the supplied slice.' % (expected, type(self).__name__))
        for i, pane in zip(range(start, end), panes):
            new_objects[i] = panel(pane)
        self.objects = new_objects

    def clone(self, *objects: Any, **params: Any) -> 'ListLike':
        """
        Makes a copy of the layout sharing the same parameters.

        Arguments
        ---------
        objects: Objects to add to the cloned layout.
        params: Keyword arguments override the parameters on the clone.

        Returns
        -------
        Cloned layout object
        """
        if not objects:
            if 'objects' in params:
                objects = params.pop('objects')
            else:
                objects = self.objects
        elif 'objects' in params:
            raise ValueError("A %s's objects should be supplied either as arguments or as a keyword, not both." % type(self).__name__)
        p = dict(self.param.values(), **params)
        del p['objects']
        return type(self)(*objects, **p)

    def append(self, obj: Any) -> None:
        """
        Appends an object to the layout.

        Arguments
        ---------
        obj (object): Panel component to add to the layout.
        """
        from ..pane import panel
        new_objects = list(self)
        new_objects.append(panel(obj))
        self.objects = new_objects

    def clear(self) -> List[Viewable]:
        """
        Clears the objects on this layout.

        Returns
        -------
        objects (list[Viewable]): List of cleared objects.
        """
        objects = self.objects
        self.objects = []
        return objects

    def extend(self, objects: Iterable[Any]) -> None:
        """
        Extends the objects on this layout with a list.

        Arguments
        ---------
        objects (list): List of panel components to add to the layout.
        """
        from ..pane import panel
        new_objects = list(self)
        new_objects.extend(list(map(panel, objects)))
        self.objects = new_objects

    def index(self, object) -> int:
        """
        Returns the integer index of the supplied object in the list of objects.

        Arguments
        ---------
        obj (object): Panel component to look up the index for.

        Returns
        -------
        index (int): Integer index of the object in the layout.
        """
        return self.objects.index(object)

    def insert(self, index: int, obj: Any) -> None:
        """
        Inserts an object in the layout at the specified index.

        Arguments
        ---------
        index (int): Index at which to insert the object.
        object (object): Panel components to insert in the layout.
        """
        from ..pane import panel
        new_objects = list(self)
        new_objects.insert(index, panel(obj))
        self.objects = new_objects

    def pop(self, index: int) -> Viewable:
        """
        Pops an item from the layout by index.

        Arguments
        ---------
        index (int): The index of the item to pop from the layout.
        """
        new_objects = list(self)
        obj = new_objects.pop(index)
        self.objects = new_objects
        return obj

    def remove(self, obj: Viewable) -> None:
        """
        Removes an object from the layout.

        Arguments
        ---------
        obj (object): The object to remove from the layout.
        """
        new_objects = list(self)
        new_objects.remove(obj)
        self.objects = new_objects

    def reverse(self) -> None:
        """
        Reverses the objects in the layout.
        """
        new_objects = list(self)
        new_objects.reverse()
        self.objects = new_objects