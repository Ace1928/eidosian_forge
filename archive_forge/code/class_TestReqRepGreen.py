import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, have_gevent
class TestReqRepGreen(GreenTest, TestReqRep):
    pass