from zope.interface import Attribute, Interface
class EnvironmentVariableNotPermitted(ValueError):
    """Setting this environment variable in this session is not permitted."""