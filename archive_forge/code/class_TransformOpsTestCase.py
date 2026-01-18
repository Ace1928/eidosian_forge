import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
class TransformOpsTestCase(unittest.TestCase):

    def test_rotate(self):
        ls = load_wkt('LINESTRING(240 400, 240 300, 300 300)')
        rls = affinity.rotate(ls, 90)
        els = load_wkt('LINESTRING(220 320, 320 320, 320 380)')
        assert rls.equals(els)
        rls = affinity.rotate(geom=ls, angle=90, origin='center')
        assert rls.equals(els)
        rls = affinity.rotate(ls, -pi / 2, use_radians=True)
        els = load_wkt('LINESTRING(320 380, 220 380, 220 320)')
        assert rls.equals(els)
        rls = affinity.rotate(ls, 90, origin='centroid')
        els = load_wkt('LINESTRING(182.5 320, 282.5 320, 282.5 380)')
        assert rls.equals(els)
        rls = affinity.rotate(ls, 90, origin=ls.coords[1])
        els = load_wkt('LINESTRING(140 300, 240 300, 240 360)')
        assert rls.equals(els)
        rls = affinity.rotate(ls, 90, origin=Point(0, 0))
        els = load_wkt('LINESTRING(-400 240, -300 240, -300 300)')
        assert rls.equals(els)

    def test_rotate_empty(self):
        rls = affinity.rotate(load_wkt('LINESTRING EMPTY'), 90)
        els = load_wkt('LINESTRING EMPTY')
        assert rls.equals(els)

    def test_rotate_angle_array(self):
        ls = load_wkt('LINESTRING(240 400, 240 300, 300 300)')
        els = load_wkt('LINESTRING(220 320, 320 320, 320 380)')
        theta = np.array(90.0)
        rls = affinity.rotate(ls, theta)
        assert theta.item() == 90.0
        assert rls.equals(els)
        theta = np.array(pi / 2)
        rls = affinity.rotate(ls, theta, use_radians=True)
        assert theta.item() == pi / 2
        assert rls.equals(els)

    def test_scale(self):
        ls = load_wkt('LINESTRING(240 400 10, 240 300 30, 300 300 20)')
        sls = affinity.scale(ls)
        assert sls.equals(ls)
        sls = affinity.scale(ls, 2, 3, 0.5)
        els = load_wkt('LINESTRING(210 500 5, 210 200 15, 330 200 10)')
        assert sls.equals(els)
        for a, b in zip(sls.coords, els.coords):
            for ap, bp in zip(a, b):
                self.assertEqual(ap, bp)
        sls = affinity.scale(geom=ls, xfact=2, yfact=3, zfact=0.5, origin='center')
        assert sls.equals(els)
        sls = affinity.scale(ls, 2, 3, 0.5, origin='centroid')
        els = load_wkt('LINESTRING(228.75 537.5, 228.75 237.5, 348.75 237.5)')
        assert sls.equals(els)
        sls = affinity.scale(ls, 2, 3, 0.5, origin=ls.coords[1])
        els = load_wkt('LINESTRING(240 600, 240 300, 360 300)')
        assert sls.equals(els)
        sls = affinity.scale(ls, 2, 3, 0.5, origin=Point(100, 200, 1000))
        els = load_wkt('LINESTRING(380 800 505, 380 500 515, 500 500 510)')
        assert sls.equals(els)
        for a, b in zip(sls.coords, els.coords):
            for ap, bp in zip(a, b):
                assert ap == bp

    def test_scale_empty(self):
        sls = affinity.scale(load_wkt('LINESTRING EMPTY'))
        els = load_wkt('LINESTRING EMPTY')
        assert sls.equals(els)

    def test_skew(self):
        ls = load_wkt('LINESTRING(240 400 10, 240 300 30, 300 300 20)')
        sls = affinity.skew(ls)
        assert sls.equals(ls)
        sls = affinity.skew(ls, 15, -30)
        els = load_wkt('LINESTRING (253.39745962155615 417.3205080756888, 226.60254037844385 317.3205080756888, 286.60254037844385 282.67949192431126)')
        assert sls.equals_exact(els, 1e-06)
        sls = affinity.skew(ls, pi / 12, -pi / 6, use_radians=True)
        assert sls.equals_exact(els, 1e-06)
        sls = affinity.skew(geom=ls, xs=15, ys=-30, origin='center', use_radians=False)
        assert sls.equals_exact(els, 1e-06)
        sls = affinity.skew(ls, 15, -30, origin='centroid')
        els = load_wkt('LINESTRING(258.42150697963973 406.49519052838332, 231.6265877365273980 306.4951905283833185, 291.6265877365274264 271.8541743770057337)')
        assert sls.equals_exact(els, 1e-06)
        sls = affinity.skew(ls, 15, -30, origin=ls.coords[1])
        els = load_wkt('LINESTRING(266.7949192431123038 400, 240 300, 300 265.3589838486224153)')
        assert sls.equals_exact(els, 1e-06)
        sls = affinity.skew(ls, 15, -30, origin=Point(0, 0))
        els = load_wkt('LINESTRING(347.179676972449101 261.435935394489832, 320.3847577293367976 161.4359353944898317, 380.3847577293367976 126.7949192431122754)')
        assert sls.equals_exact(els, 1e-06)

    def test_skew_empty(self):
        sls = affinity.skew(load_wkt('LINESTRING EMPTY'))
        els = load_wkt('LINESTRING EMPTY')
        assert sls.equals(els)

    def test_skew_xs_ys_array(self):
        ls = load_wkt('LINESTRING(240 400 10, 240 300 30, 300 300 20)')
        els = load_wkt('LINESTRING (253.39745962155615 417.3205080756888, 226.60254037844385 317.3205080756888, 286.60254037844385 282.67949192431126)')
        xs_ys = np.array([15.0, -30.0])
        sls = affinity.skew(ls, xs_ys[0, ...], xs_ys[1, ...])
        assert xs_ys[0] == 15.0
        assert xs_ys[1] == -30.0
        assert sls.equals_exact(els, 1e-06)
        xs_ys = np.array([pi / 12, -pi / 6])
        sls = affinity.skew(ls, xs_ys[0, ...], xs_ys[1, ...], use_radians=True)
        assert xs_ys[0] == pi / 12
        assert xs_ys[1] == -pi / 6
        assert sls.equals_exact(els, 1e-06)

    def test_translate(self):
        ls = load_wkt('LINESTRING(240 400 10, 240 300 30, 300 300 20)')
        tls = affinity.translate(ls)
        assert tls.equals(ls)
        tls = affinity.translate(ls, 100, 400, -10)
        els = load_wkt('LINESTRING(340 800 0, 340 700 20, 400 700 10)')
        assert tls.equals(els)
        for a, b in zip(tls.coords, els.coords):
            for ap, bp in zip(a, b):
                assert ap == bp
        tls = affinity.translate(geom=ls, xoff=100, yoff=400, zoff=-10)
        assert tls.equals(els)

    def test_translate_empty(self):
        tls = affinity.translate(load_wkt('LINESTRING EMPTY'))
        els = load_wkt('LINESTRING EMPTY')
        self.assertTrue(tls.equals(els))
        assert tls.equals(els)