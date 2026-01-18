import textwrap
import warnings
class AliasUsageWarning(Warning):
    """Use of historical service-type aliases is discouraged."""
    details = '\n    Requested service_type {given} is an old alias. Please update your\n    code to reference the official service_type {official}.\n    '