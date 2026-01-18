from fontTools.ttLib.tables import otTables as ot
from .table_builder import TableUnbuilder
def _unbuildPaintColrLayers(self, source):
    assert source['Format'] == ot.PaintFormat.PaintColrLayers
    layers = list(_flatten_layers([self.unbuildPaint(childPaint) for childPaint in self.layers[source['FirstLayerIndex']:source['FirstLayerIndex'] + source['NumLayers']]]))
    if len(layers) == 1:
        return layers[0]
    return {'Format': source['Format'], 'Layers': layers}