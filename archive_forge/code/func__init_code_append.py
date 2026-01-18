from reportlab.pdfgen import pdfgeom
from reportlab.lib.rl_accel import fp_str
def _init_code_append(self, c):
    assert c.endswith(' m') or c.endswith(' re'), 'path must start with a moveto or rect'
    code_append = self._code.append
    code_append('n')
    code_append(c)
    self._code_append = code_append