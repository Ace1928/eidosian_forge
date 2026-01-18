class AbstractResolver(object):
    """The thing that performs the actual resolution work."""
    base_exception = Exception

    def __init__(self, provider, reporter):
        self.provider = provider
        self.reporter = reporter

    def resolve(self, requirements, **kwargs):
        """Take a collection of constraints, spit out the resolution result.

        This returns a representation of the final resolution state, with one
        guarenteed attribute ``mapping`` that contains resolved candidates as
        values. The keys are their respective identifiers.

        :param requirements: A collection of constraints.
        :param kwargs: Additional keyword arguments that subclasses may accept.

        :raises: ``self.base_exception`` or its subclass.
        """
        raise NotImplementedError