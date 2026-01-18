import os
import argparse
import logging
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.ttLib import TTFont
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
import fontTools
def _font_to_cubic(input_path, output_path=None, **kwargs):
    font = TTFont(input_path)
    logger.info('Converting curves for %s', input_path)
    stats = {} if kwargs['dump_stats'] else None
    qu2cu_kwargs = {'stats': stats, 'max_err': kwargs['max_err_em'] * font['head'].unitsPerEm, 'all_cubic': kwargs['all_cubic']}
    assert 'gvar' not in font, 'Cannot convert variable font'
    glyphSet = font.getGlyphSet()
    glyphOrder = font.getGlyphOrder()
    glyf = font['glyf']
    for glyphName in glyphOrder:
        glyph = glyphSet[glyphName]
        ttpen = TTGlyphPen(glyphSet)
        pen = Qu2CuPen(ttpen, **qu2cu_kwargs)
        glyph.draw(pen)
        glyf[glyphName] = ttpen.glyph(dropImpliedOnCurves=True)
    font['head'].glyphDataFormat = 1
    if kwargs['dump_stats']:
        logger.info('Stats: %s', stats)
    logger.info('Saving %s', output_path)
    font.save(output_path)