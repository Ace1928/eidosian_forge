from ray.util.annotations import PublicAPI
class _TuneNoNextExecutorEventError(_SubCategoryTuneError):
    """Error that happens when waiting to get the next event to
    handle from RayTrialExecutor.

    Note: RayTaskError will be raised by itself and will not be using
    this category. This category is for everything else."""
    pass