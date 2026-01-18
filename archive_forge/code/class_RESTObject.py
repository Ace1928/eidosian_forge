import copy
import importlib
import json
import pprint
import textwrap
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Type, TYPE_CHECKING, Union
import gitlab
from gitlab import types as g_types
from gitlab.exceptions import GitlabParsingError
from .client import Gitlab, GitlabList
class RESTObject:
    """Represents an object built from server data.

    It holds the attributes know from the server, and the updated attributes in
    another. This allows smart updates, if the object allows it.

    You can redefine ``_id_attr`` in child classes to specify which attribute
    must be used as the unique ID. ``None`` means that the object can be updated
    without ID in the url.

    Likewise, you can define a ``_repr_attr`` in subclasses to specify which
    attribute should be added as a human-readable identifier when called in the
    object's ``__repr__()`` method.
    """
    _id_attr: Optional[str] = 'id'
    _attrs: Dict[str, Any]
    _created_from_list: bool
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _repr_attr: Optional[str] = None
    _updated_attrs: Dict[str, Any]
    _lazy: bool
    manager: 'RESTManager'

    def __init__(self, manager: 'RESTManager', attrs: Dict[str, Any], *, created_from_list: bool=False, lazy: bool=False) -> None:
        if not isinstance(attrs, dict):
            raise GitlabParsingError(f'Attempted to initialize RESTObject with a non-dictionary value: {attrs!r}\nThis likely indicates an incorrect or malformed server response.')
        self.__dict__.update({'manager': manager, '_attrs': attrs, '_updated_attrs': {}, '_module': importlib.import_module(self.__module__), '_created_from_list': created_from_list, '_lazy': lazy})
        self.__dict__['_parent_attrs'] = self.manager.parent_attrs
        self._create_managers()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        module = state.pop('_module')
        state['_module_name'] = module.__name__
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        module_name = state.pop('_module_name')
        self.__dict__.update(state)
        self.__dict__['_module'] = importlib.import_module(module_name)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__['_updated_attrs']:
            return self.__dict__['_updated_attrs'][name]
        if name in self.__dict__['_attrs']:
            value = self.__dict__['_attrs'][name]
            if isinstance(value, list):
                self.__dict__['_updated_attrs'][name] = value[:]
                return self.__dict__['_updated_attrs'][name]
            return value
        if name in self.__dict__['_parent_attrs']:
            return self.__dict__['_parent_attrs'][name]
        message = f'{type(self).__name__!r} object has no attribute {name!r}'
        if self._created_from_list:
            message = f'{message}\n\n' + textwrap.fill(f'{self.__class__!r} was created via a list() call and only a subset of the data may be present. To ensure all data is present get the object using a get(object.id) call. For more details, see:') + f'\n\n{_URL_ATTRIBUTE_ERROR}'
        elif self._lazy:
            message = f'{message}\n\n' + textwrap.fill(f'If you tried to access object attributes returned from the server, note that {self.__class__!r} was created as a `lazy` object and was not initialized with any data.')
        raise AttributeError(message)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__['_updated_attrs'][name] = value

    def asdict(self, *, with_parent_attrs: bool=False) -> Dict[str, Any]:
        data = {}
        if with_parent_attrs:
            data.update(copy.deepcopy(self._parent_attrs))
        data.update(copy.deepcopy(self._attrs))
        data.update(copy.deepcopy(self._updated_attrs))
        return data

    @property
    def attributes(self) -> Dict[str, Any]:
        return self.asdict(with_parent_attrs=True)

    def to_json(self, *, with_parent_attrs: bool=False, **kwargs: Any) -> str:
        return json.dumps(self.asdict(with_parent_attrs=with_parent_attrs), **kwargs)

    def __str__(self) -> str:
        return f'{type(self)} => {self.asdict()}'

    def pformat(self) -> str:
        return f'{type(self)} => \n{pprint.pformat(self.asdict())}'

    def pprint(self) -> None:
        print(self.pformat())

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if (self._id_attr and self._repr_value) and self._id_attr != self._repr_attr:
            return f'<{name} {self._id_attr}:{self.get_id()} {self._repr_attr}:{self._repr_value}>'
        if self._id_attr:
            return f'<{name} {self._id_attr}:{self.get_id()}>'
        if self._repr_value:
            return f'<{name} {self._repr_attr}:{self._repr_value}>'
        return f'<{name}>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RESTObject):
            return NotImplemented
        if self.get_id() and other.get_id():
            return self.get_id() == other.get_id()
        return super() == other

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, RESTObject):
            return NotImplemented
        if self.get_id() and other.get_id():
            return self.get_id() != other.get_id()
        return super() != other

    def __dir__(self) -> Iterable[str]:
        return set(self.attributes).union(super().__dir__())

    def __hash__(self) -> int:
        if not self.get_id():
            return super().__hash__()
        return hash(self.get_id())

    def _create_managers(self) -> None:
        for attr, annotation in sorted(self.__annotations__.items()):
            if attr in ('manager',):
                continue
            if not isinstance(annotation, (type, str)):
                continue
            if isinstance(annotation, type):
                cls_name = annotation.__name__
            else:
                cls_name = annotation
            if cls_name == 'RESTManager' or not cls_name.endswith('Manager'):
                continue
            cls = getattr(self._module, cls_name)
            manager = cls(self.manager.gitlab, parent=self)
            self.__dict__[attr] = manager

    def _update_attrs(self, new_attrs: Dict[str, Any]) -> None:
        self.__dict__['_updated_attrs'] = {}
        self.__dict__['_attrs'] = new_attrs

    def get_id(self) -> Optional[Union[int, str]]:
        """Returns the id of the resource."""
        if self._id_attr is None or not hasattr(self, self._id_attr):
            return None
        id_val = getattr(self, self._id_attr)
        if TYPE_CHECKING:
            assert id_val is None or isinstance(id_val, (int, str))
        return id_val

    @property
    def _repr_value(self) -> Optional[str]:
        """Safely returns the human-readable resource name if present."""
        if self._repr_attr is None or not hasattr(self, self._repr_attr):
            return None
        repr_val = getattr(self, self._repr_attr)
        if TYPE_CHECKING:
            assert isinstance(repr_val, str)
        return repr_val

    @property
    def encoded_id(self) -> Optional[Union[int, str]]:
        """Ensure that the ID is url-encoded so that it can be safely used in a URL
        path"""
        obj_id = self.get_id()
        if isinstance(obj_id, str):
            obj_id = gitlab.utils.EncodedId(obj_id)
        return obj_id