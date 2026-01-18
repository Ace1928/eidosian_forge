import random
import time
from breezy import tests, timestamp
from breezy.osutils import local_time_offset
class UnpackHighresDateTests(tests.TestCase):

    def test_unpack_highres_date(self):
        self.assertEqual((1120153132.35085, -18000), timestamp.unpack_highres_date('Thu 2005-06-30 12:38:52.350850105 -0500'))
        self.assertEqual((1120153132.35085, 0), timestamp.unpack_highres_date('Thu 2005-06-30 17:38:52.350850105 +0000'))
        self.assertEqual((1120153132.35085, 7200), timestamp.unpack_highres_date('Thu 2005-06-30 19:38:52.350850105 +0200'))
        self.assertEqual((1152428738.867522, 19800), timestamp.unpack_highres_date('Sun 2006-07-09 12:35:38.867522001 +0530'))

    def test_random(self):
        t = time.time()
        o = local_time_offset()
        t2, o2 = timestamp.unpack_highres_date(timestamp.format_highres_date(t, o))
        self.assertEqual(t, t2)
        self.assertEqual(o, o2)
        t -= 24 * 3600 * 365 * 2
        o = -12 * 3600
        for count in range(500):
            t += random.random() * 24 * 3600 * 30
            try:
                time.gmtime(t + o)
            except (OverflowError, ValueError):
                break
            if time.localtime(t).tm_year > 9998:
                break
            o = ((o / 3600 + 13) % 25 - 12) * 3600
            date = timestamp.format_highres_date(t, o)
            t2, o2 = timestamp.unpack_highres_date(date)
            self.assertEqual(t, t2, 'Failed on date {!r}, {},{} diff:{}'.format(date, t, o, t2 - t))
            self.assertEqual(o, o2, 'Failed on date {!r}, {},{} diff:{}'.format(date, t, o, t2 - t))