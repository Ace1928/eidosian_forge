from click import FileError
class NodataShadowWarning(UserWarning):
    """Warn that a dataset's nodata attribute is shadowing its alpha band."""

    def __str__(self):
        return "The dataset's nodata attribute is shadowing the alpha band. All masks will be determined by the nodata attribute"