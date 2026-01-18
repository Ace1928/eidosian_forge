import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _fill_logical(self, context, opa, opb):
    dif = context.prec - len(opa)
    if dif > 0:
        opa = '0' * dif + opa
    elif dif < 0:
        opa = opa[-context.prec:]
    dif = context.prec - len(opb)
    if dif > 0:
        opb = '0' * dif + opb
    elif dif < 0:
        opb = opb[-context.prec:]
    return (opa, opb)