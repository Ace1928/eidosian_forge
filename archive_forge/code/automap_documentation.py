from ``Engineer`` to ``Employee``, we need to set up both the relationship
from __future__ import annotations
import dataclasses
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import backref
from ..orm import declarative_base as _declarative_base
from ..orm import exc as orm_exc
from ..orm import interfaces
from ..orm import relationship
from ..orm.decl_base import _DeferredMapperConfig
from ..orm.mapper import _CONFIGURE_MUTEX
from ..schema import ForeignKeyConstraint
from ..sql import and_
from ..util import Properties
from ..util.typing import Protocol
Extract mapped classes and relationships from the
        :class:`_schema.MetaData` and perform mappings.

        For full documentation and examples see
        :ref:`automap_basic_use`.

        :param autoload_with: an :class:`_engine.Engine` or
         :class:`_engine.Connection` with which
         to perform schema reflection; when specified, the
         :meth:`_schema.MetaData.reflect` method will be invoked within
         the scope of this method.

        :param engine: legacy; use :paramref:`.AutomapBase.autoload_with`.
         Used to indicate the :class:`_engine.Engine` or
         :class:`_engine.Connection` with which to reflect tables with,
         if :paramref:`.AutomapBase.reflect` is True.

        :param reflect: legacy; use :paramref:`.AutomapBase.autoload_with`.
         Indicates that :meth:`_schema.MetaData.reflect` should be invoked.

        :param classname_for_table: callable function which will be used to
         produce new class names, given a table name.  Defaults to
         :func:`.classname_for_table`.

        :param modulename_for_table: callable function which will be used to
         produce the effective ``__module__`` for an internally generated
         class, to allow for multiple classes of the same name in a single
         automap base which would be in different "modules".

         Defaults to ``None``, which will indicate that ``__module__`` will not
         be set explicitly; the Python runtime will use the value
         ``sqlalchemy.ext.automap`` for these classes.

         When assigning ``__module__`` to generated classes, they can be
         accessed based on dot-separated module names using the
         :attr:`.AutomapBase.by_module` collection.   Classes that have
         an explicit ``__module_`` assigned using this hook do **not** get
         placed into the :attr:`.AutomapBase.classes` collection, only
         into :attr:`.AutomapBase.by_module`.

         .. versionadded:: 2.0

         .. seealso::

            :ref:`automap_by_module`

        :param name_for_scalar_relationship: callable function which will be
         used to produce relationship names for scalar relationships.  Defaults
         to :func:`.name_for_scalar_relationship`.

        :param name_for_collection_relationship: callable function which will
         be used to produce relationship names for collection-oriented
         relationships.  Defaults to :func:`.name_for_collection_relationship`.

        :param generate_relationship: callable function which will be used to
         actually generate :func:`_orm.relationship` and :func:`.backref`
         constructs.  Defaults to :func:`.generate_relationship`.

        :param collection_class: the Python collection class that will be used
         when a new :func:`_orm.relationship`
         object is created that represents a
         collection.  Defaults to ``list``.

        :param schema: Schema name to reflect when reflecting tables using
         the :paramref:`.AutomapBase.prepare.autoload_with` parameter. The name
         is passed to the :paramref:`_schema.MetaData.reflect.schema` parameter
         of :meth:`_schema.MetaData.reflect`. When omitted, the default schema
         in use by the database connection is used.

         .. note:: The :paramref:`.AutomapBase.prepare.schema`
            parameter supports reflection of a single schema at a time.
            In order to include tables from many schemas, use
            multiple calls to :meth:`.AutomapBase.prepare`.

            For an overview of multiple-schema automap including the use
            of additional naming conventions to resolve table name
            conflicts, see the section :ref:`automap_by_module`.

            .. versionadded:: 2.0 :meth:`.AutomapBase.prepare` supports being
               directly invoked any number of times, keeping track of tables
               that have already been processed to avoid processing them
               a second time.

        :param reflection_options: When present, this dictionary of options
         will be passed to :meth:`_schema.MetaData.reflect`
         to supply general reflection-specific options like ``only`` and/or
         dialect-specific options like ``oracle_resolve_synonyms``.

         .. versionadded:: 1.4

        