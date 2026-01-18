from enum import Enum
class ExportErrorType(Enum):
    INVALID_INPUT_TYPE = 1
    INVALID_OUTPUT_TYPE = 2
    VIOLATION_OF_SPEC = 3
    NOT_SUPPORTED = 4
    MISSING_PROPERTY = 5
    UNINITIALIZED = 6