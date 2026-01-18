from collections import namedtuple
import codecs
@staticmethod
def _rl_codecs(name):
    name = name.lower()
    from reportlab.pdfbase.pdfmetrics import standardEncodings
    for e in standardEncodings + ('ExtPdfdocEncoding',):
        e = e[:-8].lower()
        if name.startswith(e):
            return RL_Codecs.__rl_codecs(e)
    if name in RL_Codecs.__rl_dynamic_codecs:
        return RL_Codecs.__rl_codecs(name, _256=False)
    return None