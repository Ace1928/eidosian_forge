import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
class TestTaggedList(object):

    def test__add__(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        tl = tl + tl
        assert len(tl) == 6
        assert tl.tags == ('a', 'b', 'c', 'a', 'b', 'c')
        assert tuple(tl) == (1, 2, 3, 1, 2, 3)

    def test__delitem__(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        assert len(tl) == 3
        del tl[1]
        assert len(tl) == 2
        assert tl.tags == ('a', 'c')
        assert tuple(tl) == (1, 3)

    def test__delslice__(self):
        tl = rlc.TaggedList((1, 2, 3, 4), tags=('a', 'b', 'c', 'd'))
        del tl[1:3]
        assert len(tl) == 2
        assert tl.tags == ('a', 'd')
        assert tuple(tl) == (1, 4)

    def test__iadd__(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        tl += tl
        assert len(tl) == 6
        assert tl.tags == ('a', 'b', 'c', 'a', 'b', 'c')
        assert tuple(tl) == (1, 2, 3, 1, 2, 3)

    def test__imul__(self):
        tl = rlc.TaggedList((1, 2), tags=('a', 'b'))
        tl *= 3
        assert len(tl) == 6
        assert tl.tags == ('a', 'b', 'a', 'b', 'a', 'b')
        assert tuple(tl) == (1, 2, 1, 2, 1, 2)

    def test__init__(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        with pytest.raises(ValueError):
            rlc.TaggedList((1, 2, 3), tags=('b', 'c'))

    def test__setslice__(self):
        tl = rlc.TaggedList((1, 2, 3, 4), tags=('a', 'b', 'c', 'd'))
        tl[1:3] = [5, 6]
        assert len(tl) == 4
        assert tl.tags == ('a', 'b', 'c', 'd')
        assert tuple(tl) == (1, 5, 6, 4)

    def test_append(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        assert len(tl) == 3
        tl.append(4, tag='a')
        assert len(tl) == 4
        assert tl[3] == 4
        assert tl.tags == ('a', 'b', 'c', 'a')

    def test_extend(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        tl.extend([4, 5])
        assert tuple(tl.itertags()) == ('a', 'b', 'c', None, None)
        assert tuple(tl) == (1, 2, 3, 4, 5)

    def test_insert(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        tl.insert(1, 4, tag='d')
        assert tuple(tl.itertags()) == ('a', 'd', 'b', 'c')
        assert tuple(tl) == (1, 4, 2, 3)

    def test_items(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        assert tuple(tl.items()) == (('a', 1), ('b', 2), ('c', 3))

    def test_iterontag(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'a'))
        assert tuple(tl.iterontag('a')) == (1, 3)

    def test_itertags(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        assert tuple(tl.itertags()) == ('a', 'b', 'c')

    def test_pop(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        assert len(tl) == 3
        elt = tl.pop()
        assert elt == 3
        assert len(tl) == 2
        assert tl.tags == ('a', 'b')
        assert tuple(tl) == (1, 2)
        elt = tl.pop(0)
        assert elt == 1
        assert len(tl) == 1
        assert tl.tags == ('b',)

    def test_remove(self):
        tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
        assert len(tl) == 3
        tl.remove(2)
        assert len(tl) == 2
        assert tl.tags == ('a', 'c')
        assert tuple(tl) == (1, 3)

    def test_reverse(self):
        tn = ['a', 'b', 'c']
        tv = [1, 2, 3]
        tl = rlc.TaggedList(tv, tags=tn)
        tl.reverse()
        assert len(tl) == 3
        assert tl.tags == ('c', 'b', 'a')
        assert tuple(tl) == (3, 2, 1)

    def test_sort(self):
        tn = ['a', 'c', 'b']
        tv = [1, 3, 2]
        tl = rlc.TaggedList(tv, tags=tn)
        tl.sort()
        assert tl.tags == ('a', 'b', 'c')
        assert tuple(tl) == (1, 2, 3)

    def test_tags(self):
        tn = ['a', 'b', 'c']
        tv = [1, 2, 3]
        tl = rlc.TaggedList(tv, tags=tn)
        tags = tl.tags
        assert isinstance(tags, tuple) is True
        assert tags == ('a', 'b', 'c')
        tn = ['d', 'e', 'f']
        tl.tags = tn
        assert isinstance(tags, tuple) is True
        assert tl.tags == tuple(tn)

    def test_settag(self):
        tn = ['a', 'b', 'c']
        tv = [1, 2, 3]
        tl = rlc.TaggedList(tv, tags=tn)
        tl.settag(1, 'z')
        assert tl.tags == ('a', 'z', 'c')

    def test_from_items(self):
        od = rlc.OrdDict((('a', 1), ('b', 2), ('c', 3)))
        tl = rlc.TaggedList.from_items(od)
        assert tl.tags == ('a', 'b', 'c')
        assert tuple(tl) == (1, 2, 3)
        tl = rlc.TaggedList.from_items({'a': 1, 'b': 2, 'c': 3})
        assert set(tl.tags) == set(('a', 'b', 'c'))
        assert set(tuple(tl)) == set((1, 2, 3))

    def test_pickle(self):
        tl = rlc.TaggedList([1, 2, 3])
        tl_pickled = pickle.dumps(tl)
        tl_unpickled = pickle.loads(tl_pickled)
        assert tuple(tl) == tuple(tl_unpickled)
        assert tuple(tl.itertags()) == tuple(tl_unpickled.itertags())