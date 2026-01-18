import numpy as np
from bokeh.models import CustomJS, Toolbar
from bokeh.models.tools import RangeTool
from ...core.util import isscalar
from ..links import (
from ..plot import GenericElementPlot, GenericOverlayPlot
class DataLinkCallback(LinkCallback):
    """
    Merges the source and target ColumnDataSource
    """

    def __init__(self, root_model, link, source_plot, target_plot):
        src_cds = source_plot.handles['source']
        tgt_cds = target_plot.handles['source']
        if src_cds is tgt_cds:
            return
        src_len = [len(v) for v in src_cds.data.values()]
        tgt_len = [len(v) for v in tgt_cds.data.values()]
        if src_len and tgt_len and (src_len[0] != tgt_len[0]):
            raise ValueError('DataLink source data length must match target data length, found source length of %d and target length of %d.' % (src_len[0], tgt_len[0]))
        for k, v in tgt_cds.data.items():
            if k not in src_cds.data:
                continue
            v = np.asarray(v)
            col = np.asarray(src_cds.data[k])
            if len(v) and isinstance(v[0], np.ndarray):
                continue
            if not (isscalar(v) and v == col or (v.dtype.kind not in 'iufc' and (v == col).all()) or np.allclose(v, np.asarray(src_cds.data[k]), equal_nan=True)):
                raise ValueError('DataLink can only be applied if overlapping dimension values are equal, %s column on source does not match target' % k)
        src_cds.data.update(tgt_cds.data)
        renderer = target_plot.handles.get('glyph_renderer')
        if renderer is None:
            pass
        elif 'data_source' in renderer.properties():
            renderer.update(data_source=src_cds)
        else:
            renderer.update(source=src_cds)
        target_plot.handles['source'] = src_cds
        target_plot.handles['cds'] = src_cds
        for callback in target_plot.callbacks:
            callback.initialize(plot_id=root_model.ref['id'])