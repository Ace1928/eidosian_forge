from click import FileError
class ShapeSkipWarning(UserWarning):
    """Warn that an invalid or empty shape in a collection has been skipped"""