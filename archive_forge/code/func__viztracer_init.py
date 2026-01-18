import warnings
def _viztracer_init(init_kwargs):
    """Initialize viztracer's profiler in worker processes"""
    from viztracer import VizTracer
    tracer = VizTracer(**init_kwargs)
    tracer.register_exit()
    tracer.start()