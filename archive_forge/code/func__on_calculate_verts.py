import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase
def _on_calculate_verts(self):
    self.u_interval = self.intervals[0]
    self.u_set = list(self.u_interval.frange())
    self.v_interval = self.intervals[1]
    self.v_set = list(self.v_interval.frange())
    self.bounds = [[S.Infinity, S.NegativeInfinity, 0], [S.Infinity, S.NegativeInfinity, 0], [S.Infinity, S.NegativeInfinity, 0]]
    evaluate = self._get_evaluator()
    self._calculating_verts_pos = 0.0
    self._calculating_verts_len = float(self.u_interval.v_len * self.v_interval.v_len)
    verts = []
    b = self.bounds
    for u in self.u_set:
        column = []
        for v in self.v_set:
            try:
                _e = evaluate(u, v)
            except ZeroDivisionError:
                _e = None
            if _e is not None:
                for axis in range(3):
                    b[axis][0] = min([b[axis][0], _e[axis]])
                    b[axis][1] = max([b[axis][1], _e[axis]])
            column.append(_e)
            self._calculating_verts_pos += 1.0
        verts.append(column)
    for axis in range(3):
        b[axis][2] = b[axis][1] - b[axis][0]
        if b[axis][2] == 0.0:
            b[axis][2] = 1.0
    self.verts = verts
    self.push_wireframe(self.draw_verts(False, False))
    self.push_solid(self.draw_verts(False, True))