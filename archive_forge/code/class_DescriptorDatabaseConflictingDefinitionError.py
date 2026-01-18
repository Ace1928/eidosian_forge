import warnings
class DescriptorDatabaseConflictingDefinitionError(Error):
    """Raised when a proto is added with the same name & different descriptor."""