import unittest
from wsme import WSRoot
from wsme.protocol import getprotocol, CallContext, Protocol
import wsme.protocol
class TestProtocols(unittest.TestCase):

    def test_register_protocol(self):
        wsme.protocol.register_protocol(DummyProtocol)
        assert wsme.protocol.registered_protocols['dummy'] == DummyProtocol
        r = WSRoot()
        assert len(r.protocols) == 0
        r.addprotocol('dummy')
        assert len(r.protocols) == 1
        assert r.protocols[0].__class__ == DummyProtocol
        r = WSRoot(['dummy'])
        assert len(r.protocols) == 1
        assert r.protocols[0].__class__ == DummyProtocol

    def test_Protocol(self):
        p = wsme.protocol.Protocol()
        assert p.iter_calls(None) is None
        assert p.extract_path(None) is None
        assert p.read_arguments(None) is None
        assert p.encode_result(None, None) is None
        assert p.encode_sample_value(None, None) == ('none', 'N/A')
        assert p.encode_sample_params(None) == ('none', 'N/A')
        assert p.encode_sample_result(None, None) == ('none', 'N/A')