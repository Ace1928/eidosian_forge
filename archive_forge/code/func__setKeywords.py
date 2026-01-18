from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
def _setKeywords(self, **kwd):
    for k, v in kwd.items():
        setattr(self, k, v)