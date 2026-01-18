class NoRegionError(BaseEndpointResolverError):
    """No region was specified."""
    fmt = 'You must specify a region.'