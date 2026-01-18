import numpy
def _set_header_prec(self, prec):
    if prec in 'hilq':
        self._header_prec = prec
    else:
        raise ValueError('Cannot set header precision')