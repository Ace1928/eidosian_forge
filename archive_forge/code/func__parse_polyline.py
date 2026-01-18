import re
def _parse_polyline(self, poly):
    if 'points' in poly.attrib:
        self._start_path('M' + poly.attrib['points'])