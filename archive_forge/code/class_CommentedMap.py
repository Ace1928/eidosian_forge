from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
class CommentedMap(ordereddict, CommentedBase):
    __slots__ = (Comment.attrib, '_ok', '_ref')

    def __init__(self, *args, **kw):
        self._ok = set()
        self._ref = []
        ordereddict.__init__(self, *args, **kw)

    def _yaml_add_comment(self, comment, key=NoComment, value=NoComment):
        """values is set to key to indicate a value attachment of comment"""
        if key is not NoComment:
            self.yaml_key_comment_extend(key, comment)
            return
        if value is not NoComment:
            self.yaml_value_comment_extend(value, comment)
        else:
            self.ca.comment = comment

    def _yaml_add_eol_comment(self, comment, key):
        """add on the value line, with value specified by the key"""
        self._yaml_add_comment(comment, value=key)

    def _yaml_get_columnX(self, key):
        return self.ca.items[key][2].start_mark.column

    def _yaml_get_column(self, key):
        column = None
        sel_idx = None
        pre, post, last = (None, None, None)
        for x in self:
            if pre is not None and x != key:
                post = x
                break
            if x == key:
                pre = last
            last = x
        if pre in self.ca.items:
            sel_idx = pre
        elif post in self.ca.items:
            sel_idx = post
        else:
            for k1 in self:
                if k1 >= key:
                    break
                if k1 not in self.ca.items:
                    continue
                sel_idx = k1
        if sel_idx is not None:
            column = self._yaml_get_columnX(sel_idx)
        return column

    def _yaml_get_pre_comment(self):
        pre_comments = []
        if self.ca.comment is None:
            self.ca.comment = [None, pre_comments]
        else:
            self.ca.comment[1] = pre_comments
        return pre_comments

    def update(self, vals):
        try:
            ordereddict.update(self, vals)
        except TypeError:
            for x in vals:
                self[x] = vals[x]
        try:
            self._ok.update(vals.keys())
        except AttributeError:
            for x in vals:
                self._ok.add(x[0])

    def insert(self, pos, key, value, comment=None):
        """insert key value into given position
        attach comment if provided
        """
        ordereddict.insert(self, pos, key, value)
        self._ok.add(key)
        if comment is not None:
            self.yaml_add_eol_comment(comment, key=key)

    def mlget(self, key, default=None, list_ok=False):
        """multi-level get that expects dicts within dicts"""
        if not isinstance(key, list):
            return self.get(key, default)

        def get_one_level(key_list, level, d):
            if not list_ok:
                assert isinstance(d, dict)
            if level >= len(key_list):
                if level > len(key_list):
                    raise IndexError
                return d[key_list[level - 1]]
            return get_one_level(key_list, level + 1, d[key_list[level - 1]])
        try:
            return get_one_level(key, 1, self)
        except KeyError:
            return default
        except (TypeError, IndexError):
            if not list_ok:
                raise
            return default

    def __getitem__(self, key):
        try:
            return ordereddict.__getitem__(self, key)
        except KeyError:
            for merged in getattr(self, merge_attrib, []):
                if key in merged[1]:
                    return merged[1][key]
            raise

    def __setitem__(self, key, value):
        if key in self:
            if isinstance(value, string_types) and (not isinstance(value, ScalarString)) and isinstance(self[key], ScalarString):
                value = type(self[key])(value)
        ordereddict.__setitem__(self, key, value)
        self._ok.add(key)

    def _unmerged_contains(self, key):
        if key in self._ok:
            return True
        return None

    def __contains__(self, key):
        return bool(ordereddict.__contains__(self, key))

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except:
            return default

    def __repr__(self):
        return ordereddict.__repr__(self).replace('CommentedMap', 'ordereddict')

    def non_merged_items(self):
        for x in ordereddict.__iter__(self):
            if x in self._ok:
                yield (x, ordereddict.__getitem__(self, x))

    def __delitem__(self, key):
        self._ok.discard(key)
        ordereddict.__delitem__(self, key)
        for referer in self._ref:
            referer.update_key_value(key)

    def __iter__(self):
        for x in ordereddict.__iter__(self):
            yield x

    def _keys(self):
        for x in ordereddict.__iter__(self):
            yield x

    def __len__(self):
        return ordereddict.__len__(self)

    def __eq__(self, other):
        return bool(dict(self) == other)
    if PY2:

        def keys(self):
            return list(self._keys())

        def iterkeys(self):
            return self._keys()

        def viewkeys(self):
            return CommentedMapKeysView(self)
    else:

        def keys(self):
            return CommentedMapKeysView(self)
    if PY2:

        def _values(self):
            for x in ordereddict.__iter__(self):
                yield ordereddict.__getitem__(self, x)

        def values(self):
            return list(self._values())

        def itervalues(self):
            return self._values()

        def viewvalues(self):
            return CommentedMapValuesView(self)
    else:

        def values(self):
            return CommentedMapValuesView(self)

    def _items(self):
        for x in ordereddict.__iter__(self):
            yield (x, ordereddict.__getitem__(self, x))
    if PY2:

        def items(self):
            return list(self._items())

        def iteritems(self):
            return self._items()

        def viewitems(self):
            return CommentedMapItemsView(self)
    else:

        def items(self):
            return CommentedMapItemsView(self)

    @property
    def merge(self):
        if not hasattr(self, merge_attrib):
            setattr(self, merge_attrib, [])
        return getattr(self, merge_attrib)

    def copy(self):
        x = type(self)()
        for k, v in self._items():
            x[k] = v
        self.copy_attributes(x)
        return x

    def add_referent(self, cm):
        if cm not in self._ref:
            self._ref.append(cm)

    def add_yaml_merge(self, value):
        for v in value:
            v[1].add_referent(self)
            for k, v in v[1].items():
                if ordereddict.__contains__(self, k):
                    continue
                ordereddict.__setitem__(self, k, v)
        self.merge.extend(value)

    def update_key_value(self, key):
        if key in self._ok:
            return
        for v in self.merge:
            if key in v[1]:
                ordereddict.__setitem__(self, key, v[1][key])
                return
        ordereddict.__delitem__(self, key)

    def __deepcopy__(self, memo):
        res = self.__class__()
        memo[id(self)] = res
        for k in self:
            res[k] = copy.deepcopy(self[k])
        self.copy_attributes(res, deep=True)
        return res