class HidesTestsPydoctorSystem(System):
    """A PyDoctor "system" used to generate the docs."""

    def privacyClass(self, documentable: Documentable) -> PrivacyClass:
        """Report the privacy level for an object. Hide the module
            'docstring_parser.tests'.
            """
        if documentable.fullName().startswith('docstring_parser.tests'):
            return PrivacyClass.HIDDEN
        return super().privacyClass(documentable)