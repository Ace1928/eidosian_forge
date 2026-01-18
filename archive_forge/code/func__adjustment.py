import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _adjustment(self, adjustment):
    adv, dx, dy, adv_adjust_by, dx_adjust_by, dy_adjust_by = adjustment
    adv_device = adv_adjust_by and adv_adjust_by.items() or None
    dx_device = dx_adjust_by and dx_adjust_by.items() or None
    dy_device = dy_adjust_by and dy_adjust_by.items() or None
    return ast.ValueRecord(xPlacement=dx, yPlacement=dy, xAdvance=adv, xPlaDevice=dx_device, yPlaDevice=dy_device, xAdvDevice=adv_device)