from . import compat
class InvalidComponentsError(ValidationError):
    """Exception raised when one or more components are invalid."""

    def __init__(self, uri, *component_names):
        """Initialize the error with the invalid component name(s)."""
        verb = 'was'
        if len(component_names) > 1:
            verb = 'were'
        self.uri = uri
        self.components = sorted(component_names)
        components = ', '.join(self.components)
        super(InvalidComponentsError, self).__init__('{} {} found to be invalid'.format(components, verb), uri, self.components)