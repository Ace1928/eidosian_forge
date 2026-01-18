def IsCloudPath(path):
    """Checks whether a given path is Cloud filesystem path."""
    return path.startswith('gs://') or path.startswith('s3://')