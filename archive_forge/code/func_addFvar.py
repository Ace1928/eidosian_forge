from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from .ttLib import TTFont, newTable
from .ttLib.tables._c_m_a_p import cmap_classes
from .ttLib.tables._g_l_y_f import flagCubic
from .ttLib.tables.O_S_2f_2 import Panose
from .misc.timeTools import timestampNow
import struct
from collections import OrderedDict
def addFvar(font, axes, instances):
    from .ttLib.tables._f_v_a_r import Axis, NamedInstance
    assert axes
    fvar = newTable('fvar')
    nameTable = font['name']
    for axis_def in axes:
        axis = Axis()
        if isinstance(axis_def, tuple):
            axis.axisTag, axis.minValue, axis.defaultValue, axis.maxValue, name = axis_def
        else:
            axis.axisTag, axis.minValue, axis.defaultValue, axis.maxValue, name = (axis_def.tag, axis_def.minimum, axis_def.default, axis_def.maximum, axis_def.name)
            if axis_def.hidden:
                axis.flags = 1
        if isinstance(name, str):
            name = dict(en=name)
        axis.axisNameID = nameTable.addMultilingualName(name, ttFont=font)
        fvar.axes.append(axis)
    for instance in instances:
        if isinstance(instance, dict):
            coordinates = instance['location']
            name = instance['stylename']
            psname = instance.get('postscriptfontname')
        else:
            coordinates = instance.location
            name = instance.localisedStyleName or instance.styleName
            psname = instance.postScriptFontName
        if isinstance(name, str):
            name = dict(en=name)
        inst = NamedInstance()
        inst.subfamilyNameID = nameTable.addMultilingualName(name, ttFont=font)
        if psname is not None:
            inst.postscriptNameID = nameTable.addName(psname)
        inst.coordinates = coordinates
        fvar.instances.append(inst)
    font['fvar'] = fvar