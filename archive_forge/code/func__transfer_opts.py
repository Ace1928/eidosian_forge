from holoviews.core.overlay import CompositeOverlay
from holoviews.core.options import Store
from holoviews.plotting.util import COLOR_ALIASES
def _transfer_opts(element, backend):
    """
    Transfer the bokeh options of an element to another backend
    based on an internal mapping of option transforms.
    """
    elname = type(element).__name__
    options = Store.options(backend=backend)
    transforms = BACKEND_TRANSFORMS[backend]
    if isinstance(element, CompositeOverlay):
        element = element.apply(_transfer_opts, backend=backend, per_element=True)
    new_opts = {}
    el_options = element.opts.get(backend='bokeh', defaults=False).kwargs
    for grp, el_opts in options[elname].groups.items():
        for opt, val in el_options.items():
            transform = transforms.get(grp, {}).get(opt, None)
            if transform is None and _is_interactive_opt(opt):
                transform = UNSET
            if transform is UNSET:
                continue
            elif transform:
                opt, val = transform(opt, val)
                if val is UNSET:
                    continue
            if opt not in el_opts.allowed_keywords:
                continue
            new_opts[opt] = val
    if backend == 'matplotlib':
        size_opts = _transform_size_to_mpl(el_options.get('width'), el_options.get('height'), el_options.get('aspect'))
        new_opts.update(size_opts)
    return element.opts(**new_opts, backend=backend)