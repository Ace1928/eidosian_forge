import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
class TestHTTPHeaderDict(object):

    def test_create_from_kwargs(self):
        h = HTTPHeaderDict(ab=1, cd=2, ef=3, gh=4)
        assert len(h) == 4
        assert 'ab' in h

    def test_create_from_dict(self):
        h = HTTPHeaderDict(dict(ab=1, cd=2, ef=3, gh=4))
        assert len(h) == 4
        assert 'ab' in h

    def test_create_from_iterator(self):
        teststr = 'urllib3ontherocks'
        h = HTTPHeaderDict(((c, c * 5) for c in teststr))
        assert len(h) == len(set(teststr))

    def test_create_from_list(self):
        headers = [('ab', 'A'), ('cd', 'B'), ('cookie', 'C'), ('cookie', 'D'), ('cookie', 'E')]
        h = HTTPHeaderDict(headers)
        assert len(h) == 3
        assert 'ab' in h
        clist = h.getlist('cookie')
        assert len(clist) == 3
        assert clist[0] == 'C'
        assert clist[-1] == 'E'

    def test_create_from_headerdict(self):
        headers = [('ab', 'A'), ('cd', 'B'), ('cookie', 'C'), ('cookie', 'D'), ('cookie', 'E')]
        org = HTTPHeaderDict(headers)
        h = HTTPHeaderDict(org)
        assert len(h) == 3
        assert 'ab' in h
        clist = h.getlist('cookie')
        assert len(clist) == 3
        assert clist[0] == 'C'
        assert clist[-1] == 'E'
        assert h is not org
        assert h == org

    def test_setitem(self, d):
        d['Cookie'] = 'foo'
        assert d['cookie'] == 'foo'
        d['cookie'] = 'with, comma'
        assert d.getlist('cookie') == ['with, comma']

    def test_update(self, d):
        d.update(dict(Cookie='foo'))
        assert d['cookie'] == 'foo'
        d.update(dict(cookie='with, comma'))
        assert d.getlist('cookie') == ['with, comma']

    def test_delitem(self, d):
        del d['cookie']
        assert 'cookie' not in d
        assert 'COOKIE' not in d

    def test_add_well_known_multiheader(self, d):
        d.add('COOKIE', 'asdf')
        assert d.getlist('cookie') == ['foo', 'bar', 'asdf']
        assert d['cookie'] == 'foo, bar, asdf'

    def test_add_comma_separated_multiheader(self, d):
        d.add('bar', 'foo')
        d.add('BAR', 'bar')
        d.add('Bar', 'asdf')
        assert d.getlist('bar') == ['foo', 'bar', 'asdf']
        assert d['bar'] == 'foo, bar, asdf'

    def test_extend_from_list(self, d):
        d.extend([('set-cookie', '100'), ('set-cookie', '200'), ('set-cookie', '300')])
        assert d['set-cookie'] == '100, 200, 300'

    def test_extend_from_dict(self, d):
        d.extend(dict(cookie='asdf'), b='100')
        assert d['cookie'] == 'foo, bar, asdf'
        assert d['b'] == '100'
        d.add('cookie', 'with, comma')
        assert d.getlist('cookie') == ['foo', 'bar', 'asdf', 'with, comma']

    def test_extend_from_container(self, d):
        h = NonMappingHeaderContainer(Cookie='foo', e='foofoo')
        d.extend(h)
        assert d['cookie'] == 'foo, bar, foo'
        assert d['e'] == 'foofoo'
        assert len(d) == 2

    def test_extend_from_headerdict(self, d):
        h = HTTPHeaderDict(Cookie='foo', e='foofoo')
        d.extend(h)
        assert d['cookie'] == 'foo, bar, foo'
        assert d['e'] == 'foofoo'
        assert len(d) == 2

    @pytest.mark.parametrize('args', [(1, 2), (1, 2, 3, 4, 5)])
    def test_extend_with_wrong_number_of_args_is_typeerror(self, d, args):
        with pytest.raises(TypeError) as err:
            d.extend(*args)
        assert 'extend() takes at most 1 positional arguments' in err.value.args[0]

    def test_copy(self, d):
        h = d.copy()
        assert d is not h
        assert d == h

    def test_getlist(self, d):
        assert d.getlist('cookie') == ['foo', 'bar']
        assert d.getlist('Cookie') == ['foo', 'bar']
        assert d.getlist('b') == []
        d.add('b', 'asdf')
        assert d.getlist('b') == ['asdf']

    def test_getlist_after_copy(self, d):
        assert d.getlist('cookie') == HTTPHeaderDict(d).getlist('cookie')

    def test_equal(self, d):
        b = HTTPHeaderDict(cookie='foo, bar')
        c = NonMappingHeaderContainer(cookie='foo, bar')
        assert d == b
        assert d == c
        assert d != 2

    def test_not_equal(self, d):
        b = HTTPHeaderDict(cookie='foo, bar')
        c = NonMappingHeaderContainer(cookie='foo, bar')
        assert not d != b
        assert not d != c
        assert d != 2

    def test_pop(self, d):
        key = 'Cookie'
        a = d[key]
        b = d.pop(key)
        assert a == b
        assert key not in d
        with pytest.raises(KeyError):
            d.pop(key)
        dummy = object()
        assert dummy is d.pop(key, dummy)

    def test_discard(self, d):
        d.discard('cookie')
        assert 'cookie' not in d
        d.discard('cookie')

    def test_len(self, d):
        assert len(d) == 1
        d.add('cookie', 'bla')
        d.add('asdf', 'foo')
        assert len(d) == 2

    def test_repr(self, d):
        rep = "HTTPHeaderDict({'Cookie': 'foo, bar'})"
        assert repr(d) == rep

    def test_items(self, d):
        items = d.items()
        assert len(items) == 2
        assert items[0][0] == 'Cookie'
        assert items[0][1] == 'foo'
        assert items[1][0] == 'Cookie'
        assert items[1][1] == 'bar'

    def test_dict_conversion(self, d):
        hdict = {'Content-Length': '0', 'Content-type': 'text/plain', 'Server': 'TornadoServer/1.2.3'}
        h = dict(HTTPHeaderDict(hdict).items())
        assert hdict == h
        assert hdict == dict(HTTPHeaderDict(hdict))

    def test_string_enforcement(self, d):
        with pytest.raises(Exception):
            d[3] = 5
        with pytest.raises(Exception):
            d.add(3, 4)
        with pytest.raises(Exception):
            del d[3]
        with pytest.raises(Exception):
            HTTPHeaderDict({3: 3})

    @pytest.mark.skipif(not six.PY2, reason='python3 has a different internal header implementation')
    def test_from_httplib_py2(self):
        msg = '\nServer: nginx\nContent-Type: text/html; charset=windows-1251\nConnection: keep-alive\nX-Some-Multiline: asdf\n asdf\t\n\t asdf\nSet-Cookie: bb_lastvisit=1348253375; expires=Sat, 21-Sep-2013 18:49:35 GMT; path=/\nSet-Cookie: bb_lastactivity=0; expires=Sat, 21-Sep-2013 18:49:35 GMT; path=/\nwww-authenticate: asdf\nwww-authenticate: bla\n\n'
        buffer = six.moves.StringIO(msg.lstrip().replace('\n', '\r\n'))
        msg = six.moves.http_client.HTTPMessage(buffer)
        d = HTTPHeaderDict.from_httplib(msg)
        assert d['server'] == 'nginx'
        cookies = d.getlist('set-cookie')
        assert len(cookies) == 2
        assert cookies[0].startswith('bb_lastvisit')
        assert cookies[1].startswith('bb_lastactivity')
        assert d['x-some-multiline'] == 'asdf asdf asdf'
        assert d['www-authenticate'] == 'asdf, bla'
        assert d.getlist('www-authenticate') == ['asdf', 'bla']
        with_invalid_multiline = '\tthis-is-not-a-header: but it has a pretend value\nAuthorization: Bearer 123\n\n'
        buffer = six.moves.StringIO(with_invalid_multiline.replace('\n', '\r\n'))
        msg = six.moves.http_client.HTTPMessage(buffer)
        with pytest.raises(InvalidHeader):
            HTTPHeaderDict.from_httplib(msg)