from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class AliasConflictParameterError(ValidationError):
    """
    Error when an alias is provided for a parameter as well as the original.

    :ivar original: The name of the original parameter.
    :ivar alias: The name of the alias
    :ivar operation: The name of the operation.
    """
    fmt = "Parameter '{original}' and its alias '{alias}' were provided for operation {operation}.  Only one of them may be used."