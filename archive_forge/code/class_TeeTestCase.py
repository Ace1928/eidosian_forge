import sys
from io import StringIO
import unittest
from IPython.utils.io import Tee, capture_output
class TeeTestCase(unittest.TestCase):

    def tchan(self, channel):
        trap = StringIO()
        chan = StringIO()
        text = 'Hello'
        std_ori = getattr(sys, channel)
        setattr(sys, channel, trap)
        tee = Tee(chan, channel=channel)
        print(text, end='', file=chan)
        trap_val = trap.getvalue()
        self.assertEqual(chan.getvalue(), text)
        tee.close()
        setattr(sys, channel, std_ori)
        assert getattr(sys, channel) == std_ori

    def test(self):
        for chan in ['stdout', 'stderr']:
            self.tchan(chan)