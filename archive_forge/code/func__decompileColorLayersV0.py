from fontTools.misc.textTools import safeEval
from . import DefaultTable
@staticmethod
def _decompileColorLayersV0(table):
    if not table.LayerRecordArray:
        return {}
    colorLayerLists = {}
    layerRecords = table.LayerRecordArray.LayerRecord
    numLayerRecords = len(layerRecords)
    for baseRec in table.BaseGlyphRecordArray.BaseGlyphRecord:
        baseGlyph = baseRec.BaseGlyph
        firstLayerIndex = baseRec.FirstLayerIndex
        numLayers = baseRec.NumLayers
        assert firstLayerIndex + numLayers <= numLayerRecords
        layers = []
        for i in range(firstLayerIndex, firstLayerIndex + numLayers):
            layerRec = layerRecords[i]
            layers.append(LayerRecord(layerRec.LayerGlyph, layerRec.PaletteIndex))
        colorLayerLists[baseGlyph] = layers
    return colorLayerLists