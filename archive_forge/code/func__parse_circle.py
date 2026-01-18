import re
def _parse_circle(self, circle):
    cx = float(circle.attrib.get('cx', 0))
    cy = float(circle.attrib.get('cy', 0))
    r = float(circle.attrib.get('r'))
    self._start_path()
    self.M(cx - r, cy)
    self.A(r, r, cx + r, cy, large_arc=1)
    self.A(r, r, cx - r, cy, large_arc=1)