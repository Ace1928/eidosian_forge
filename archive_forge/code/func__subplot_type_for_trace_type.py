import collections
def _subplot_type_for_trace_type(trace_type):
    from plotly.validators import DataValidator
    trace_validator = DataValidator()
    if trace_type in trace_validator.class_strs_map:
        trace = trace_validator.validate_coerce([{'type': trace_type}])[0]
        if 'domain' in trace:
            return 'domain'
        elif 'xaxis' in trace and 'yaxis' in trace:
            return 'xy'
        elif 'geo' in trace:
            return 'geo'
        elif 'scene' in trace:
            return 'scene'
        elif 'subplot' in trace:
            for t in _subplot_prop_named_subplot:
                try:
                    trace.subplot = t
                    return t
                except ValueError:
                    pass
    return None