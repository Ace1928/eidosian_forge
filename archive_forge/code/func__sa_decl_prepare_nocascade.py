from __future__ import annotations
import collections
import contextlib
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import Union
from ... import exc as sa_exc
from ...engine import Connection
from ...engine import Engine
from ...orm import exc as orm_exc
from ...orm import relationships
from ...orm.base import _mapper_or_none
from ...orm.clsregistry import _resolver
from ...orm.decl_base import _DeferredMapperConfig
from ...orm.util import polymorphic_union
from ...schema import Table
from ...util import OrderedDict
@classmethod
def _sa_decl_prepare_nocascade(cls):
    if getattr(cls, '__mapper__', None):
        return
    to_map = _DeferredMapperConfig.config_for_cls(cls)
    mappers = []
    stack = list(cls.__subclasses__())
    while stack:
        klass = stack.pop()
        stack.extend(klass.__subclasses__())
        mn = _mapper_or_none(klass)
        if mn is not None:
            mappers.append(mn)
    discriminator_name = getattr(cls, '_concrete_discriminator_name', None) or 'type'
    pjoin = cls._create_polymorphic_union(mappers, discriminator_name)
    declared_cols = set(to_map.declared_columns)
    declared_col_keys = {c.key for c in declared_cols}
    for k, v in list(to_map.properties.items()):
        if v in declared_cols:
            to_map.properties[k] = pjoin.c[v.key]
            declared_col_keys.remove(v.key)
    to_map.local_table = pjoin
    strict_attrs = cls.__dict__.get('strict_attrs', False)
    m_args = to_map.mapper_args_fn or dict

    def mapper_args():
        args = m_args()
        args['polymorphic_on'] = pjoin.c[discriminator_name]
        args['polymorphic_abstract'] = True
        if strict_attrs:
            args['include_properties'] = set(pjoin.primary_key) | declared_col_keys | {discriminator_name}
            args['with_polymorphic'] = ('*', pjoin)
        return args
    to_map.mapper_args_fn = mapper_args
    to_map.map()
    stack = [cls]
    while stack:
        scls = stack.pop(0)
        stack.extend(scls.__subclasses__())
        sm = _mapper_or_none(scls)
        if sm and sm.concrete and (sm.inherits is None):
            for sup_ in scls.__mro__[1:]:
                sup_sm = _mapper_or_none(sup_)
                if sup_sm:
                    sm._set_concrete_base(sup_sm)
                    break