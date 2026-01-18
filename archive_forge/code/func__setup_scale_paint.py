from fontTools.ttLib.ttVisitor import TTVisitor
import fontTools.ttLib as ttLib
import fontTools.ttLib.tables.otBase as otBase
import fontTools.ttLib.tables.otTables as otTables
from fontTools.cffLib import VarStoreData
import fontTools.cffLib.specializer as cffSpecializer
from fontTools.varLib import builder  # for VarData.calculateNumShorts
from fontTools.misc.fixedTools import otRound
from fontTools.ttLib.tables._g_l_y_f import VarComponentFlags
def _setup_scale_paint(paint, scale):
    if -2 <= scale <= 2 - (1 >> 14):
        paint.Format = otTables.PaintFormat.PaintScaleUniform
        paint.scale = scale
        return
    transform = otTables.Affine2x3()
    transform.populateDefaults()
    transform.xy = transform.yx = transform.dx = transform.dy = 0
    transform.xx = transform.yy = scale
    paint.Format = otTables.PaintFormat.PaintTransform
    paint.Transform = transform