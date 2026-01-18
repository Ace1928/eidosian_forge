class OSError(DefaultCredentialsError, EnvironmentError):
    """Used to wrap EnvironmentError(OSError after python3.3)."""