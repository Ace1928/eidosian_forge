from . import compat
class UnpermittedComponentError(ValidationError):
    """Exception raised when a component has an unpermitted value."""

    def __init__(self, component_name, component_value, allowed_values):
        """Initialize the error with the unpermitted component."""
        super(UnpermittedComponentError, self).__init__('{} was required to be one of {!r} but was {!r}'.format(component_name, list(sorted(allowed_values)), component_value), component_name, component_value, allowed_values)
        self.component_name = component_name
        self.component_value = component_value
        self.allowed_values = allowed_values