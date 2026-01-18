from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def _errorDump(fontName, fontSize):
    s1, s2 = list(map(str, exc_info()[:2]))
    from reportlab import rl_config
    if rl_config.verbose >= 2:
        import os
        _ = os.path.join(os.path.dirname(rl_config.__file__), 'fonts')
        print('!!!!! %s: %s' % (_, os.listdir(_)))
        for _ in ('T1SearchPath', 'TTFSearchPath'):
            print('!!!!! rl_config.%s = %s' % (_, repr(getattr(rl_config, _))))
    code = 'raise RenderPMError("Error in setFont(%s,%s) missing the T1 files?\\nOriginally %s: %s")' % (repr(fontName), repr(fontSize), s1, s2)
    code += ' from None'
    rl_exec(code, dict(RenderPMError=RenderPMError))