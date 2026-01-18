import os
import sys
import time
from pytest import mark
import zmq
from zmq.tests import GreenTest, PollZMQTestCase, have_gevent
class TestSelect(PollZMQTestCase):

    def test_pair(self):
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        wait()
        rlist, wlist, xlist = zmq.select([s1, s2], [s1, s2], [s1, s2])
        assert s1 in wlist
        assert s2 in wlist
        assert s1 not in rlist
        assert s2 not in rlist

    @mark.flaky(reruns=3)
    def test_timeout(self):
        """make sure select timeout has the right units (seconds)."""
        s1, s2 = self.create_bound_pair(zmq.PAIR, zmq.PAIR)
        tic = time.perf_counter()
        r, w, x = zmq.select([s1, s2], [], [], 0.005)
        toc = time.perf_counter()
        assert toc - tic < 1
        assert toc - tic > 0.001
        tic = time.perf_counter()
        r, w, x = zmq.select([s1, s2], [], [], 0.25)
        toc = time.perf_counter()
        assert toc - tic < 1
        assert toc - tic > 0.1