from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import strategy_options
from .base import _DeclarativeMapped
from .base import class_mapper
from .descriptor_props import CompositeProperty
from .descriptor_props import ConcreteInheritedProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .interfaces import StrategizedProperty
from .relationships import RelationshipProperty
from .util import de_stringify_annotation
from .util import de_stringify_union_elements
from .. import exc as sa_exc
from .. import ForeignKey
from .. import log
from .. import util
from ..sql import coercions
from ..sql import roles
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import SchemaConst
from ..sql.type_api import TypeEngine
from ..util.typing import de_optionalize_union_types
from ..util.typing import is_fwd_ref
from ..util.typing import is_optional_union
from ..util.typing import is_pep593
from ..util.typing import is_pep695
from ..util.typing import is_union
from ..util.typing import Self
from ..util.typing import typing_get_args
def _orm_annotate_column(self, column: _NC) -> _NC:
    """annotate and possibly adapt a column to be returned
            as the mapped-attribute exposed version of the column.

            The column in this context needs to act as much like the
            column in an ORM mapped context as possible, so includes
            annotations to give hints to various ORM functions as to
            the source entity of this column.   It also adapts it
            to the mapper's with_polymorphic selectable if one is
            present.

            """
    pe = self._parententity
    annotations: Dict[str, Any] = {'entity_namespace': pe, 'parententity': pe, 'parentmapper': pe, 'proxy_key': self.prop.key}
    col = column
    if self._parentmapper._polymorphic_adapter:
        mapper_local_col = col
        col = self._parentmapper._polymorphic_adapter.traverse(col)
        annotations['adapt_column'] = mapper_local_col
    return col._annotate(annotations)._set_propagate_attrs({'compile_state_plugin': 'orm', 'plugin_subject': pe})