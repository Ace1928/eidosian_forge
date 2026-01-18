import collections
def _get_subplot_ref_for_trace(trace):
    if 'domain' in trace:
        return SubplotRef(subplot_type='domain', layout_keys=(), trace_kwargs={'domain': {'x': trace.domain.x, 'y': trace.domain.y}})
    elif 'xaxis' in trace and 'yaxis' in trace:
        xaxis_name = 'xaxis' + trace.xaxis[1:] if trace.xaxis else 'xaxis'
        yaxis_name = 'yaxis' + trace.yaxis[1:] if trace.yaxis else 'yaxis'
        return SubplotRef(subplot_type='xy', layout_keys=(xaxis_name, yaxis_name), trace_kwargs={'xaxis': trace.xaxis, 'yaxis': trace.yaxis})
    elif 'geo' in trace:
        return SubplotRef(subplot_type='geo', layout_keys=(trace.geo,), trace_kwargs={'geo': trace.geo})
    elif 'scene' in trace:
        return SubplotRef(subplot_type='scene', layout_keys=(trace.scene,), trace_kwargs={'scene': trace.scene})
    elif 'subplot' in trace:
        for t in _subplot_prop_named_subplot:
            try:
                validator = trace._get_prop_validator('subplot')
                validator.validate_coerce(t)
                return SubplotRef(subplot_type=t, layout_keys=(trace.subplot,), trace_kwargs={'subplot': trace.subplot})
            except ValueError:
                pass
    return None