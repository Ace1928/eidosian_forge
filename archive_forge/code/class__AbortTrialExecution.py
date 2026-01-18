from ray.util.annotations import PublicAPI
class _AbortTrialExecution(TuneError):
    """Error that indicates a trial should not be retried."""
    pass