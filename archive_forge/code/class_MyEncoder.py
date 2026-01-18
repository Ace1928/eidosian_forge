import datetime
import json
import numpy as np
from ase.utils import reader, writer
class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if hasattr(obj, 'todict'):
            d = obj.todict()
            if not isinstance(d, dict):
                raise RuntimeError('todict() of {} returned object of type {} but should have returned dict'.format(obj, type(d)))
            if hasattr(obj, 'ase_objtype'):
                d['__ase_objtype__'] = obj.ase_objtype
            return d
        if isinstance(obj, np.ndarray):
            flatobj = obj.ravel()
            if np.iscomplexobj(obj):
                flatobj.dtype = obj.real.dtype
            return {'__ndarray__': (obj.shape, obj.dtype.name, flatobj.tolist())}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime.datetime):
            return {'__datetime__': obj.isoformat()}
        if isinstance(obj, complex):
            return {'__complex__': (obj.real, obj.imag)}
        return json.JSONEncoder.default(self, obj)