from click import FileError
class RasterioDeprecationWarning(FutureWarning):
    """Rasterio module deprecations

    Following https://www.python.org/dev/peps/pep-0565/#additional-use-case-for-futurewarning
    we base this on FutureWarning while continuing to support Python < 3.7.
    """