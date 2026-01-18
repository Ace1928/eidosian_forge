import re
def _parse_ellipse(self, ellipse):
    cx = float(ellipse.attrib.get('cx', 0))
    cy = float(ellipse.attrib.get('cy', 0))
    rx = float(ellipse.attrib.get('rx'))
    ry = float(ellipse.attrib.get('ry'))
    self._start_path()
    self.M(cx - rx, cy)
    self.A(rx, ry, cx + rx, cy, large_arc=1)
    self.A(rx, ry, cx - rx, cy, large_arc=1)