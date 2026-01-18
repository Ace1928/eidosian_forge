import base64
import datetime
import decimal
from wsme.rest.xml import fromxml, toxml
import wsme.tests.protocol
from wsme.types import isarray, isdict, isusertype, register_type
from wsme.utils import parse_isodatetime, parse_isodate, parse_isotime
class TestRestXML(wsme.tests.protocol.RestOnlyProtocolTestCase):
    protocol = 'restxml'

    def call(self, fpath, _rt=None, _accept=None, _no_result_decode=False, body=None, **kw):
        if body:
            el = dumpxml('body', body)
        else:
            el = dumpxml('parameters', kw)
        content = et.tostring(el)
        headers = {'Content-Type': 'text/xml'}
        if _accept is not None:
            headers['Accept'] = _accept
        res = self.app.post('/' + fpath, content, headers=headers, expect_errors=True)
        print('Received:', res.body)
        if _no_result_decode:
            return res
        el = et.fromstring(res.body)
        if el.tag == 'error':
            raise wsme.tests.protocol.CallException(el.find('faultcode').text, el.find('faultstring').text, el.find('debuginfo') is not None and el.find('debuginfo').text or None)
        else:
            return loadxml(et.fromstring(res.body), _rt)

    def test_encode_sample_value(self):

        class MyType(object):
            aint = int
            atext = wsme.types.text
        register_type(MyType)
        value = MyType()
        value.aint = 5
        value.atext = 'test'
        language, sample = wsme.rest.xml.encode_sample_value(MyType, value, True)
        print(language, sample)
        assert language == 'xml'
        assert sample == b'<value>\n  <aint>5</aint>\n  <atext>test</atext>\n</value>'

    def test_encode_sample_params(self):
        lang, content = wsme.rest.xml.encode_sample_params([('a', int, 2)], True)
        assert lang == 'xml', lang
        assert content == b'<parameters>\n  <a>2</a>\n</parameters>', content

    def test_encode_sample_result(self):
        lang, content = wsme.rest.xml.encode_sample_result(int, 2, True)
        assert lang == 'xml', lang
        assert content == b'<result>2</result>', content

    def test_nil_fromxml(self):
        for dt in (str, [int], {int: str}, bool, datetime.date, datetime.time, datetime.datetime):
            e = et.Element('value', nil='true')
            assert fromxml(dt, e) is None

    def test_nil_toxml(self):
        for dt in (wsme.types.bytes, [int], {int: str}, bool, datetime.date, datetime.time, datetime.datetime):
            x = et.tostring(toxml(dt, 'value', None))
            assert x == b'<value nil="true" />', x

    def test_unset_attrs(self):

        class AType(object):
            someattr = wsme.types.bytes
        wsme.types.register_type(AType)
        x = et.tostring(toxml(AType, 'value', AType()))
        assert x == b'<value />', x