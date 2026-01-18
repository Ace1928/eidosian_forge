from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
def _calculate(self):
    self.validate()
    self.encode()
    self.decompose()
    self.computeSize()