from ray.util.annotations import PublicAPI
@PublicAPI
class TuneError(Exception):
    """General error class raised by ray.tune."""
    pass