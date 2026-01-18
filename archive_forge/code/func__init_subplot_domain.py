import collections
def _init_subplot_domain(x_domain, y_domain):
    subplot_ref = SubplotRef(subplot_type='domain', layout_keys=(), trace_kwargs={'domain': {'x': tuple(x_domain), 'y': tuple(y_domain)}})
    return (subplot_ref,)