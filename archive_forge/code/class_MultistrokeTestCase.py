import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
class MultistrokeTestCase(unittest.TestCase):

    def setUp(self):
        global best_score
        best_score = 0
        counter = 0
        self.Tinvar = MultistrokeGesture('T', [TGesture], orientation_sensitive=False)
        self.Tbound = MultistrokeGesture('T', [TGesture], orientation_sensitive=True)
        self.Ninvar = MultistrokeGesture('N', [NGesture], orientation_sensitive=False)
        self.Nbound = MultistrokeGesture('N', [NGesture], orientation_sensitive=True)

    @pytest.fixture(autouse=True)
    def set_clock(self, kivy_clock):
        self.kivy_clock = kivy_clock

    def test_immediate(self):
        gdb = Recognizer(db=[self.Tinvar, self.Ninvar])
        r = gdb.recognize([Ncandidate], max_gpf=0)
        self.assertEqual(r._match_ops, 4)
        self.assertEqual(r._completed, 2)
        self.assertEqual(r.progress, 1)
        self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)

    def test_scheduling(self):
        global best_score
        from kivy.clock import Clock
        gdb = Recognizer(db=[self.Tinvar, self.Ninvar])
        r = gdb.recognize([Ncandidate], max_gpf=1)
        r.bind(on_complete=best_score_cb)
        Clock.tick()
        self.assertEqual(r.progress, 0.5)
        self.assertEqual(best_score, 0.0)
        Clock.tick()
        self.assertEqual(r.progress, 1)
        self.assertTrue(best_score > 0.94 and best_score < 0.95)

    def test_scheduling_limits(self):
        global best_score
        from kivy.clock import Clock
        gdb = Recognizer(db=[self.Ninvar])
        tpls = len(self.Ninvar.templates)
        best_score = 0
        gdb.db.append(self.Ninvar)
        r = gdb.recognize([Ncandidate], max_gpf=1)
        r.bind(on_complete=best_score_cb)
        self.assertEqual(r.progress, 0)
        Clock.tick()
        self.assertEqual(r.progress, 0.5)
        self.assertEqual(best_score, 0)
        Clock.tick()
        self.assertEqual(r.progress, 1)
        self.assertTrue(best_score > 0.94 and best_score < 0.95)
        best_score = 0
        gdb.db.append(self.Ninvar)
        r = gdb.recognize([Ncandidate], max_gpf=1)
        r.bind(on_complete=best_score_cb)
        self.assertEqual(r.progress, 0)
        Clock.tick()
        self.assertEqual(r.progress, 1 / 3.0)
        Clock.tick()
        self.assertEqual(r.progress, 2 / 3.0)
        self.assertEqual(best_score, 0)
        Clock.tick()
        self.assertEqual(r.progress, 1)
        self.assertTrue(best_score > 0.94 and best_score < 0.95)

    def test_parallel_recognize(self):
        global counter
        from kivy.clock import Clock
        counter = 0
        gdb = Recognizer()
        for i in range(9):
            gdb.add_gesture('T', [TGesture], priority=50)
        gdb.add_gesture('N', [NGesture])
        r1 = gdb.recognize([Ncandidate], max_gpf=1)
        r1.bind(on_complete=counter_cb)
        Clock.tick()
        r2 = gdb.recognize([Ncandidate], max_gpf=1)
        r2.bind(on_complete=counter_cb)
        Clock.tick()
        r3 = gdb.recognize([Ncandidate], max_gpf=1)
        r3.bind(on_complete=counter_cb)
        Clock.tick()
        for i in range(5):
            n = gdb.recognize([TGesture], max_gpf=0)
            self.assertEqual(n.best['name'], 'T')
            self.assertTrue(round(n.best['score'], 1) == 1.0)
        for i in range(6):
            Clock.tick()
        self.assertEqual(counter, 0)
        Clock.tick()
        self.assertEqual(counter, 1)
        Clock.tick()
        self.assertEqual(counter, 2)
        Clock.tick()
        self.assertEqual(counter, 3)

    def test_timeout_case_1(self):
        global best_score
        from kivy.clock import Clock
        from time import sleep
        best_score = 0
        gdb = Recognizer(db=[self.Tbound, self.Ninvar])
        r = gdb.recognize([Ncandidate], max_gpf=1, timeout=0.4)
        Clock.tick()
        self.assertEqual(best_score, 0)
        sleep(0.4)
        Clock.tick()
        self.assertEqual(r.status, 'timeout')
        self.assertEqual(r.progress, 0.5)
        self.assertTrue(r.best['name'] == 'T')
        self.assertTrue(r.best['score'] < 0.5)

    def test_timeout_case_2(self):
        global best_score
        from kivy.clock import Clock
        from time import sleep
        best_score = 0
        gdb = Recognizer(db=[self.Tbound, self.Ninvar, self.Tinvar])
        r = gdb.recognize([Ncandidate], max_gpf=1, timeout=0.8)
        Clock.tick()
        self.assertEqual(best_score, 0)
        sleep(0.4)
        Clock.tick()
        sleep(0.4)
        Clock.tick()
        self.assertEqual(r.status, 'timeout')
        self.assertEqual(r.progress, 2 / 3.0)
        self.assertTrue(r.best['score'] >= 0.94 and r.best['score'] <= 0.95)

    def test_priority_sorting(self):
        gdb = Recognizer()
        gdb.add_gesture('N', [NGesture], priority=10)
        gdb.add_gesture('T', [TGesture], priority=5)
        r = gdb.recognize([Ncandidate], goodscore=0.01, max_gpf=0, force_priority_sort=True)
        self.assertEqual(r.best['name'], 'T')
        r = gdb.recognize([Ncandidate], goodscore=0.01, force_priority_sort=False, max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        r = gdb.recognize([Ncandidate], goodscore=0.01, max_gpf=0, priority=10)
        self.assertEqual(r.best['name'], 'T')
        r = gdb.recognize([Ncandidate], goodscore=0.01, max_gpf=0, priority=4)
        self.assertEqual(r.best['name'], None)

    def test_name_filter(self):
        gdb = Recognizer(db=[self.Ninvar, self.Nbound])
        n = gdb.filter()
        self.assertEqual(len(n), 2)
        n = gdb.filter(name='X')
        self.assertEqual(len(n), 0)

    def test_numpoints_filter(self):
        gdb = Recognizer(db=[self.Ninvar, self.Nbound])
        n = gdb.filter(numpoints=100)
        self.assertEqual(len(n), 0)
        gdb.add_gesture('T', [TGesture], numpoints=100)
        n = gdb.filter(numpoints=100)
        self.assertEqual(len(n), 1)
        n = gdb.filter(numpoints=[100, 16])
        self.assertEqual(len(n), 3)

    def test_numstrokes_filter(self):
        gdb = Recognizer(db=[self.Ninvar, self.Nbound])
        n = gdb.filter(numstrokes=2)
        self.assertEqual(len(n), 0)
        gdb.add_gesture('T', [TGesture, TGesture])
        n = gdb.filter(numstrokes=2)
        self.assertEqual(len(n), 1)
        n = gdb.filter(numstrokes=[1, 2])
        self.assertEqual(len(n), 3)

    def test_priority_filter(self):
        gdb = Recognizer(db=[self.Ninvar, self.Nbound])
        n = gdb.filter(priority=50)
        self.assertEqual(len(n), 0)
        gdb.add_gesture('T', [TGesture], priority=51)
        n = gdb.filter(priority=50)
        self.assertEqual(len(n), 0)
        n = gdb.filter(priority=51)
        self.assertEqual(len(n), 1)
        gdb.add_gesture('T', [TGesture], priority=52)
        n = gdb.filter(priority=[0, 51])
        self.assertEqual(len(n), 1)
        n = gdb.filter(priority=[0, 52])
        self.assertEqual(len(n), 2)
        n = gdb.filter(priority=[51, 52])
        self.assertEqual(len(n), 2)
        n = gdb.filter(priority=[52, 53])
        self.assertEqual(len(n), 1)
        n = gdb.filter(priority=[53, 54])
        self.assertEqual(len(n), 0)

    def test_orientation_filter(self):
        gdb = Recognizer(db=[self.Ninvar, self.Nbound])
        n = gdb.filter(orientation_sensitive=True)
        self.assertEqual(len(n), 1)
        n = gdb.filter(orientation_sensitive=False)
        self.assertEqual(len(n), 1)
        n = gdb.filter(orientation_sensitive=None)
        self.assertEqual(len(n), 2)
        gdb.db.append(self.Tinvar)
        n = gdb.filter(orientation_sensitive=True)
        self.assertEqual(len(n), 1)
        n = gdb.filter(orientation_sensitive=False)
        self.assertEqual(len(n), 2)
        n = gdb.filter(orientation_sensitive=None)
        self.assertEqual(len(n), 3)

    def test_resample(self):
        r = kivy.multistroke.resample([Vector(0, 0), Vector(1, 1)], 11)
        self.assertEqual(len(r), 11)
        self.assertEqual(round(r[9].x, 1), 0.9)
        r = kivy.multistroke.resample(TGesture, 25)
        self.assertEqual(len(r), 25)
        self.assertEqual(round(r[12].x), 81)
        self.assertEqual(r[12].y, 7)
        self.assertEqual(TGesture[3].x, r[24].x)
        self.assertEqual(TGesture[3].y, r[24].y)

    def test_rotateby(self):
        r = kivy.multistroke.rotate_by(NGesture, 24)
        self.assertEqual(round(r[2].x, 1), 158.6)
        self.assertEqual(round(r[2].y, 1), 54.9)

    def test_transfer(self):
        gdb1 = Recognizer(db=[self.Ninvar])
        gdb2 = Recognizer()
        gdb1.transfer_gesture(gdb2, name='N')
        r = gdb2.recognize([Ncandidate], max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)

    def test_export_import_case_1(self):
        gdb1 = Recognizer(db=[self.Ninvar])
        gdb2 = Recognizer()
        g = gdb1.export_gesture(name='N')
        gdb2.import_gesture(g)
        r = gdb2.recognize([Ncandidate], max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)

    def test_export_import_case_2(self):
        from tempfile import mkstemp
        import os
        gdb1 = Recognizer(db=[self.Ninvar, self.Tinvar])
        gdb2 = Recognizer()
        fh, fn = mkstemp()
        os.close(fh)
        g = gdb1.export_gesture(name='N', filename=fn)
        gdb2.import_gesture(filename=fn)
        os.unlink(fn)
        self.assertEqual(len(gdb1.db), 2)
        self.assertEqual(len(gdb2.db), 1)
        r = gdb2.recognize([Ncandidate], max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)

    def test_protractor_invariant(self):
        gdb = Recognizer(db=[self.Tinvar, self.Ninvar])
        r = gdb.recognize([NGesture], orientation_sensitive=False, max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] == 1.0)
        r = gdb.recognize([NGesture], orientation_sensitive=True, max_gpf=0)
        self.assertEqual(r.best['name'], None)
        self.assertEqual(r.best['score'], 0)
        r = gdb.recognize([Ncandidate], orientation_sensitive=False, max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)

    def test_protractor_bound(self):
        gdb = Recognizer(db=[self.Tbound, self.Nbound])
        r = gdb.recognize([NGesture], orientation_sensitive=True, max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] >= 0.99)
        r = gdb.recognize([NGesture], orientation_sensitive=False, max_gpf=0)
        self.assertEqual(r.best['name'], None)
        self.assertEqual(r.best['score'], 0)
        r = gdb.recognize([Ncandidate], orientation_sensitive=True, max_gpf=0)
        self.assertEqual(r.best['name'], 'N')
        self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)