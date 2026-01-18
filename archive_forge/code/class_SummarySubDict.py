import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
class SummarySubDict:
    """Nested dict-like object that proxies read and write operations through a root object.

    This lets us do synchronous serialization and lazy loading of large values.
    """

    def __init__(self, root=None, path=()):
        self._path = tuple(path)
        if root is None:
            self._root = self
            self._json_dict = {}
        else:
            self._root = root
            json_dict = root._json_dict
            for k in path:
                json_dict = json_dict.get(k, {})
            self._json_dict = json_dict
        self._dict = {}
        self._locked_keys = set()

    def __setattr__(self, k, v):
        k = k.strip()
        if k.startswith('_'):
            object.__setattr__(self, k, v)
        else:
            self[k] = v

    def __getattr__(self, k):
        k = k.strip()
        if k.startswith('_'):
            return object.__getattribute__(self, k)
        else:
            return self[k]

    def _root_get(self, path, child_dict):
        """Load a value at a particular path from the root.

        This should only be implemented by the "_root" child class.

        We pass the child_dict so the item can be set on it or not as
        appropriate. Returning None for a nonexistant path wouldn't be
        distinguishable from that path being set to the value None.
        """
        raise NotImplementedError

    def _root_set(self, path, new_keys_values):
        """Set a value at a particular path in the root.

        This should only be implemented by the "_root" child class.
        """
        raise NotImplementedError

    def _root_del(self, path):
        """Delete a value at a particular path in the root.

        This should only be implemented by the "_root" child class.
        """
        raise NotImplementedError

    def _write(self, commit=False):
        raise NotImplementedError

    def keys(self):
        return self._json_dict.keys()

    def get(self, k, default=None):
        if isinstance(k, str):
            k = k.strip()
        if k not in self._dict:
            self._root._root_get(self._path + (k,), self._dict)
        return self._dict.get(k, default)

    def items(self):
        for k in self.keys():
            yield (k, self[k])

    def __getitem__(self, k):
        if isinstance(k, str):
            k = k.strip()
        self.get(k)
        res = self._dict[k]
        return res

    def __contains__(self, k):
        if isinstance(k, str):
            k = k.strip()
        return k in self._json_dict

    def __setitem__(self, k, v):
        if isinstance(k, str):
            k = k.strip()
        path = self._path
        if isinstance(v, dict):
            self._dict[k] = SummarySubDict(self._root, path + (k,))
            self._root._root_set(path, [(k, {})])
            self._dict[k].update(v)
        else:
            self._dict[k] = v
            self._root._root_set(path, [(k, v)])
        self._locked_keys.add(k)
        self._root._write()
        return v

    def __delitem__(self, k):
        k = k.strip()
        del self._dict[k]
        self._root._root_del(self._path + (k,))
        self._root._write()

    def __repr__(self):
        repr_dict = dict(self._dict)
        for k in self._json_dict:
            v = self._json_dict[k]
            if k not in repr_dict and isinstance(v, dict) and (v.get('_type') in H5_TYPES):
                repr_dict[k] = '...'
            else:
                repr_dict[k] = self[k]
        return repr(repr_dict)

    def update(self, key_vals=None, overwrite=True):
        """Locked keys will be overwritten unless overwrite=False.

        Otherwise, written keys will be added to the "locked" list.
        """
        if key_vals:
            write_items = self._update(key_vals, overwrite)
            self._root._root_set(self._path, write_items)
        self._root._write(commit=True)

    def _update(self, key_vals, overwrite):
        if not key_vals:
            return
        key_vals = {k.strip(): v for k, v in key_vals.items()}
        if overwrite:
            write_items = list(key_vals.items())
            self._locked_keys.update(key_vals.keys())
        else:
            write_keys = set(key_vals.keys()) - self._locked_keys
            write_items = [(k, key_vals[k]) for k in write_keys]
        for key, value in write_items:
            if isinstance(value, dict):
                self._dict[key] = SummarySubDict(self._root, self._path + (key,))
                self._dict[key]._update(value, overwrite)
            else:
                self._dict[key] = value
        return write_items