from __future__ import absolute_import, division, print_function
class AnsibleValidationErrorMultiple(AnsibleValidationError):
    """Multiple argument spec validation errors"""

    def __init__(self, errors=None):
        self.errors = errors[:] if errors else []
        ':class:`list` of :class:`AnsibleValidationError` objects'

    def __getitem__(self, key):
        return self.errors[key]

    def __setitem__(self, key, value):
        self.errors[key] = value

    def __delitem__(self, key):
        del self.errors[key]

    @property
    def msg(self):
        """The first message from the first error in ``errors``."""
        return self.errors[0].args[0]

    @property
    def messages(self):
        """:class:`list` of each error message in ``errors``."""
        return [err.msg for err in self.errors]

    def append(self, error):
        """Append a new error to ``self.errors``.

        Only :class:`AnsibleValidationError` should be added.
        """
        self.errors.append(error)

    def extend(self, errors):
        """Append each item in ``errors`` to ``self.errors``. Only :class:`AnsibleValidationError` should be added."""
        self.errors.extend(errors)