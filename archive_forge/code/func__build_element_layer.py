from ...core.options import Store
from ...core.overlay import NdOverlay, Overlay
from ...selection import OverlaySelectionDisplay
def _build_element_layer(self, element, layer_color, layer_alpha, **opts):
    backend_options = Store.options(backend='plotly')
    style_options = backend_options[type(element).name,]['style']
    allowed = style_options.allowed_keywords
    if 'selectedpoints' in allowed:
        shared_opts = dict(selectedpoints=False)
    else:
        shared_opts = {}
    merged_opts = dict(shared_opts)
    if 'opacity' in allowed:
        merged_opts['opacity'] = layer_alpha
    elif 'alpha' in allowed:
        merged_opts['alpha'] = layer_alpha
    if layer_color is not None:
        merged_opts.update(self._get_color_kwarg(layer_color))
    else:
        for color_prop in self.color_props:
            current_color = element.opts.get(group='style')[0].get(color_prop, None)
            if current_color:
                merged_opts.update({color_prop: current_color})
    for opt in ('cmap', 'colorbar'):
        if opt in opts and opt in allowed:
            merged_opts[opt] = opts[opt]
    filtered = {k: v for k, v in merged_opts.items() if k in allowed}
    return element.opts(clone=True, backend='plotly', **filtered)