from io import BytesIO
import pickle
import re
from .. import Column
from .. import Table
from ..engine import Engine
from ..orm import class_mapper
from ..orm.interfaces import MapperProperty
from ..orm.mapper import Mapper
from ..orm.session import Session
from ..util import b64decode
from ..util import b64encode
def Serializer(*args, **kw):
    pickler = pickle.Pickler(*args, **kw)

    def persistent_id(obj):
        if isinstance(obj, Mapper) and (not obj.non_primary):
            id_ = 'mapper:' + b64encode(pickle.dumps(obj.class_))
        elif isinstance(obj, MapperProperty) and (not obj.parent.non_primary):
            id_ = 'mapperprop:' + b64encode(pickle.dumps(obj.parent.class_)) + ':' + obj.key
        elif isinstance(obj, Table):
            if 'parententity' in obj._annotations:
                id_ = 'mapper_selectable:' + b64encode(pickle.dumps(obj._annotations['parententity'].class_))
            else:
                id_ = f'table:{obj.key}'
        elif isinstance(obj, Column) and isinstance(obj.table, Table):
            id_ = f'column:{obj.table.key}:{obj.key}'
        elif isinstance(obj, Session):
            id_ = 'session:'
        elif isinstance(obj, Engine):
            id_ = 'engine:'
        else:
            return None
        return id_
    pickler.persistent_id = persistent_id
    return pickler