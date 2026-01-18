from click import FileError
class ResamplingAlgorithmError(RasterioError):
    """Raised when a resampling algorithm is invalid or inapplicable"""